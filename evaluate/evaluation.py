from torch.distributed.distributed_c10d import group
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import io
import pickle

import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import global_config as cfg 
# from config21 import global_config as cfg  # global, already initialized


import warnings
warnings.filterwarnings("ignore")


class Modal(object):
    def __init__(self, device):
        self.device = device
        # initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)

        # create model: gpt2
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  # single gpu

        #
        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None
        
        # If 'multi-gpu' is True
        if cfg.multi_gpu:   # Multi-GPU validate(2022/1/4)
            self._pickler = pickle.Pickler
            self._unpickler = pickle.Unpickler

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        ################################# 2022/1/11 ##########################
        num_training_steps = self.reader.set_stats['train']['num_training_steps_per_epoch'] * cfg.epoch_num 
        ######################################################################
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs
    
    ##### 2021/12/28 #####
    def add_torch_input_eval_batch(self, inputs, context_name):
        # inputs: context
        # Padding and return attention mask
        contexts_np, lengths, attention_mask = utils.padSeqs_gpt_pre(inputs[context_name], self.tokenizer.encode('<pad>')[0])
        
        # To tensor
        inputs[context_name + '_tensor'] = torch.LongTensor(
            contexts_np).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        return inputs, lengths, attention_mask
    ######################

    def calculate_loss_and_accuracy(self, outputs, labels): 
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]  # lm_logits: [batch_size, time_steps, vocab_size]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() # labels: [batch_size, time_steps]

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def train_URURU(self):
        """
        URURU
        """
        all_batches = self.reader.get_batches('train')
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_data_iterator(
                all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn)   # pv_batch + turn_batch
                    pv_batch = inputs['labels'] 
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            self.log_first_inputs(inputs)
                            log_inputs -= 1

                        # to tensor
                        inputs = self.add_torch_input(inputs)
                        # loss
                        outputs = self.model(inputs['contexts_tensor'])
                        # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                        loss = self.calculate_loss_and_accuracy(
                            outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        # epoch_step + 1 --> epoch_step (2021/12/20)
                        if (epoch_step) % cfg.gradient_accumulation_steps == 0 or(
                            # end of an epoch
                            (epoch_step) == set_stats['num_training_steps_per_epoch']
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1

                            logs = {}  # for tb writer
                            # logging: loss, lr... after certain amount of steps
                            if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                                loss_scalar = (tr_loss - logging_loss) / \
                                    cfg.report_interval
                                logging_loss = tr_loss
                                logs['loss'] = loss_scalar
                                logging.info(
                                    'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                        global_step, epoch_step, loss_scalar
                                    ))
                                # validate
                                # add to tensorboard...
                                if cfg.evaluate_during_training and loss_scalar < 10:
                                    results = self.validate()
                                    for k, v in results.items():
                                        eval_key = "eval_{}".format(k)
                                        logs[eval_key] = v

                                if self.tb_writer:
                                    for k, v in logs.items():
                                        self.tb_writer.add_scalar(
                                            k, v, global_step)
                                # save model... 

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))
            # save model after every epoch
            # if epoch > 30 and tr_loss/epoch_step < 0.6:
            self.save_model(epoch, tr_loss/epoch_step)




    def train(self):
        """
        UBARU
        """
        all_batches = self.reader.get_batches('train')    
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()    # 定义优化器

        # log info
        set_stats = self.reader.set_stats['train']
        if cfg.multi_gpu is None:
            logging.info("***** Running training on single GPU*****")
            logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                        set_stats['num_training_steps_per_epoch']) 
            logging.info("  Num Turns = %d", set_stats['num_turns'])
            logging.info("  Num Dialogs = %d", set_stats['num_dials'])
            logging.info("  Num Epochs = %d", cfg.epoch_num)
            logging.info("  Batch size  = %d", cfg.batch_size)
            logging.info("  Gradient Accumulation steps = %d",
                        cfg.gradient_accumulation_steps)
        else:
            if cfg.local_rank == 0:
                logging.info("***** Running training on multiple GPUs*****")
                logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                            set_stats['num_training_steps_per_epoch']) 
                logging.info("  Num Turns = %d", set_stats['num_turns'])
                logging.info("  Num Dialogs = %d", set_stats['num_dials'])
                logging.info("  Num Epochs = %d", cfg.epoch_num)
                logging.info("  Batch size  = %d", cfg.batch_size)
                logging.info("  Gradient Accumulation steps / GPU = %d",
                            cfg.gradient_accumulation_steps)
                
        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()


        # if DDP model (2021/12/18)
        if cfg.multi_gpu: # multiple gpu, use DDP
            self.model = DDP(self.model, device_ids=[self.device])
        
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0   #
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(
            all_batches) # (2021/12/20)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch) 
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    # loss
                    outputs = self.model(inputs['contexts_tensor']) # inputs['contexts_tensor'](tensor): [2, 475]; outputs[0] == logits: [2, 475, 50324]
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                    loss = self.calculate_loss_and_accuracy(
                        outputs, labels=inputs['contexts_tensor']) / cfg.gradient_accumulation_steps # 由于使用了梯度累加，应该再除以gradient_accumulation_steps(2021/12/15) --> 已修正(2021/12/20)
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    # epoch_step + 1 --> epoch_step (2021/12/20)
                    if (epoch_step) % cfg.gradient_accumulation_steps == 0 or(
                        # end of an epoch
                        (epoch_step) == set_stats['num_training_steps_per_epoch']    # "num_training_steps_per_epoch" 应该以session level计算吧？？？（2021/12/15）--> 已经修正 (2021/12/20)
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) / \
                                cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))
                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and loss_scalar < 10:
                                results = self.validate()
                                for k, v in list(results.items())[:-1]: # 'for k, v in results.items()' NameError: name 'workspace' is not defined
                                    eval_key = "eval_{}".format(k)
                                    logs[eval_key] = v

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(
                                        k, v, global_step)
                            # save model... 

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))

            # save model after every epoch
            # if epoch > 10 or tr_loss/epoch_step < 1:
            if cfg.multi_gpu and dist.get_rank() == 0: 
                self.save_model_ddp(epoch, tr_loss/epoch_step)
            if not cfg.multi_gpu:
                self.save_model(epoch, tr_loss/epoch_step)

    def train_UBARU_GEN_CONTEXT(self):
        """
        UBARU with generated context
        """
        # get list of batch
        all_batches = self.reader.get_batches('train')
        # get optimizer and scheduler
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        if cfg.multi_gpu is None:
            logging.info("***** Running training on single GPU*****")
            logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                        set_stats['num_training_steps_per_epoch']) 
            logging.info("  Num Turns = %d", set_stats['num_turns'])
            logging.info("  Num Dialogs = %d", set_stats['num_dials'])
            logging.info("  Num Epochs = %d", cfg.epoch_num)
            logging.info("  Batch size  = %d", cfg.batch_size)
            logging.info("  Gradient Accumulation steps = %d",
                        cfg.gradient_accumulation_steps)
        else:
            if cfg.local_rank == 0:
                logging.info("***** Running training on multiple GPUs*****")
                logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                            set_stats['num_training_steps_per_epoch']) 
                logging.info("  Num Turns = %d", set_stats['num_turns'])
                logging.info("  Num Dialogs = %d", set_stats['num_dials'])
                logging.info("  Num Epochs = %d", cfg.epoch_num)
                logging.info("  Batch size  = %d", cfg.batch_size)
                logging.info("  Gradient Accumulation steps / GPU = %d",
                            cfg.gradient_accumulation_steps)

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Whether use DDP model (2021/12/27)
        if cfg.multi_gpu: # multiple gpu, use DDP
            self.model = DDP(self.model, device_ids=[self.device])
        
        # Using pretrained model

                
        log_inputs = 2
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_data_iterator(
                all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs = self.reader.convert_batch_turn_gen(turn_batch, pv_batch, first_turn, cfg.gen_context)   # pv_batch + turn_batch
                    pv_batch = inputs['labels'] 
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            self.log_first_inputs(inputs)
                            log_inputs -= 1

                        # to tensor
                        inputs = self.add_torch_input(inputs)
                        # loss
                        outputs = self.model(inputs['contexts_tensor'])
                        
                        ##### update pv_batch (append generated context) 2022/1/9 #####
                        lm_logits_batch = outputs[0]
                        for i, pv, lm_logits in zip(range(len(pv_batch)), pv_batch, lm_logits_batch):
                            if cfg.gen_context == 'r':
                                sos_r_id = self.tokenizer.encode('<sos_r>')[0]
                                # get index of <sos_r> in i-th turn sample in the batch
                                sos_r_idx = torch.nonzero((inputs['contexts_tensor'][i] == sos_r_id))[-1]
                                gen_r = torch.argmax(lm_logits[sos_r_idx:-1], dim=-1) # ignore last token
                                pv = pv + [sos_r_id] + gen_r.tolist()
                                pv_batch[i] = pv
                            elif cfg.gen_context == 'bdar': # 待测试
                                sos_b_id = self.tokenizer.encode('<sos_b>')[0]
                                # get index of <sos_b> in i-th turn sample in the batch
                                sos_b_idx = torch.nonzero((inputs['contexts_tensor'][i] == sos_b_id))[-1]
                                gen_bdar = torch.argmax(lm_logits[sos_b_idx:-1], dim=-1) # ignore last token
                                pv = pv + [sos_b_id] + gen_bdar.tolist()
                                pv_batch[i] = pv
                        #######################################################

                        loss = self.calculate_loss_and_accuracy(
                            outputs, labels=inputs['contexts_tensor']) / cfg.gradient_accumulation_steps
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if (epoch_step) % cfg.gradient_accumulation_steps == 0 or(
                            # end of an epoch
                            (epoch_step) == set_stats['num_training_steps_per_epoch']    # "num_training_steps_per_epoch" 应该以session level计算吧？？？（2021/12/15）--> 已经修正 (2021/12/20)
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1

                            logs = {}  # for tb writer
                            # logging: loss, lr... after certain amount of steps
                            if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                                loss_scalar = (tr_loss - logging_loss) / \
                                    cfg.report_interval
                                logging_loss = tr_loss
                                logs['loss'] = loss_scalar
                                logging.info(
                                    'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                        global_step, epoch_step, loss_scalar
                                    ))
                                # validate
                                # add to tensorboard...
                                if cfg.evaluate_during_training and loss_scalar < 10:
                                    results = self.validate()
                                    for k, v in list(results.items())[:-1]: # 'for k, v in results.items()' NameError: name 'workspace' is not defined
                                        eval_key = "eval_{}".format(k)
                                        logs[eval_key] = v

                                if self.tb_writer:
                                    for k, v in logs.items():
                                        self.tb_writer.add_scalar(
                                            k, v, global_step)
                                # save model... 

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))

            # save model after every epoch
            # if epoch > 10 or tr_loss/epoch_step < 1:
            if cfg.multi_gpu and dist.get_rank() == 0: 
                self.save_model_ddp(epoch, tr_loss/epoch_step)
            if not cfg.multi_gpu:
                self.save_model(epoch, tr_loss/epoch_step)

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

    def save_model_ddp(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2_multi-gpu'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.module.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

    def validate_URURU(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()

        # all_batches = self.reader.get_batches('dev')
        # data_iterator = self.reader.get_data_iterator(all_batches)
        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            eval_pbar = eval_data
            for dial_idx, dialog in enumerate(eval_pbar):

                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval_URURU(
                        turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    if cfg.use_true_curr_bspn: # generate act, response
                        max_len=60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                    use_cache=cfg.use_cache)
                                                    #   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0],
                                                    use_cache=cfg.use_cache)
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        # check DB result
                        if cfg.use_true_db_pointer:
                            # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                            db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db + self.tokenizer.encode(['<sos_a>'])]).to(self.device)
                        context_length = len(inputs['context_tensor_db'][0])
                        outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                                    max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                    use_cache=cfg.use_cache)
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length-1:]
                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                    
                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']

                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    # pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    # pv_turn['db'] = db
                    # pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                    # pv_turn = inputs['labels']
                    
                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("Inference time: {:.2f} min".format((time.time()-btm)/60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('Validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
            match, success, bleu, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match

        return eval_results

    def validate(self, data='dev', do_test=False, multi_gpu=False):
        '''
        validating on single gpu or multiple gpus
        when multiple gpus, follow following steps:
            1. caculate "pred" on each rank --> DistributedSampler
            2. gather (pred and target) on rank 0
            3. stack or cat as appropriate
            4. calculate metrics（and reduce）
        '''
        # predict one dialog/ one turn at a time
        self.model.eval()

        eval_data = self.reader.get_eval_data(data)

        ######################## use for debug (2021/12/18) ############################
        #eval_data = eval_data[:5]
        ################################################################################

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            for dial_idx, dialog in enumerate(eval_data): 

                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval(
                        turn, pv_turn, first_turn) # inputs
                    inputs = self.add_torch_input_eval(inputs)

                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context']) # 
                    if cfg.use_true_curr_bspn: # generate act, response
                        max_len=60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                    use_cache=cfg.use_cache)
                                                    #   no_repeat_ngram_size=4
                        # turn['generated'] = self.tokenizer.decode(outputs[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length-1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                    max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0],
                                                    use_cache=cfg.use_cache)
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
                        # check DB result
                        if cfg.use_true_db_pointer:
                            # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
                            db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db + self.tokenizer.encode(['<sos_a>'])]).to(self.device)
                        context_length = len(inputs['context_tensor_db'][0])
                        outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                                    max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                    use_cache=cfg.use_cache)
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length-1:]
                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                    
                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']

                    # check DB resultss
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels'] # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("Inference time: {:.2f} min".format((time.time()-btm)/60))
        
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('Validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
            match, success, bleu, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)

        
        model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
        eval_on = '-'.join(cfg.exp_domains)
        if data == 'test':
            eval_on += '_test'
        if not os.path.exists(cfg.log_path):
            os.mkdir(cfg.log_path)
        log_file_name = os.path.join(cfg.log_path, model_setting+'-'+eval_on+'.json')
        if os.path.exists(log_file_name):
            eval_to_json = json.load(open(log_file_name, 'r'))
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        else:
            eval_to_json = {}
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        logging.info('Update eval results to {}'.format(log_file_name))
        return eval_results

    def validate_UBARU_GEN_CONTEXT(self, data='dev', do_test=False, multi_gpu=False):
        pass
    
    def validate_batch(self, data='dev', do_test=False, multi_gpu=False):
    
        # Predict one dialog/ one turn at a time
        self.model.eval()

        # Get batch data
        eval_data_batch = self.reader.get_batches(data) # all data list
        set_stats = self.reader.set_stats[data]

        ###### Debug 2022/3/10 #####
        # eval_data_batch = eval_data_batch[-1:]
        ############################
        
        # Logging
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            data_iterator = self.reader.get_data_iterator(
                eval_data_batch)
            for batch_idx, dial_batch in enumerate(data_iterator): 
                
                # Record previous turn. list of dict
                pv_turn_batch = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    
                    # Construct context
                    first_turn = (turn_num == 0)
                    inputs = self.reader.convert_turn_eval_batch(
                        turn_batch, pv_turn_batch, first_turn) 
                    inputs, lengths, attention_mask = self.add_torch_input_eval_batch(inputs, context_name='context')

                    # fail to generate new tokens, if max_length not set
                    context_length = inputs['context_tensor'].shape[1] 
                    if cfg.use_true_curr_bspn: # generate act, response # 该分支未测试 2021/12/31
                        max_len=60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(input_ids=inputs['context_tensor'], attention_mask=attention_mask,
                                                    max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
                                                    pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                    use_cache=cfg.use_cache)

                        # 2022-05-11
                        # Resp_gen, need to trim previous context and padded token in the end
                        # generated = outputs.cpu().numpy().tolist()   # outputs shape: [batch_size, time_step]
                        # generated = generated[context_length-1:]

                        #---------------------------------------------------------------
                        # debug 2022-05-11
                        outputs_batch = outputs.cpu().numpy()
                        generated_ar_batch = outputs_batch[:, context_length-1:].tolist()
                        try:
                            decoded_batch = self.decode_generated_act_resp_batch(generated_ar_batch, turn_batch['dial_id'], turn_num)
                            decoded_batch['bspn'] = None
                        except ValueError as exception:
                            logging.info(str(exception))
                            # logging
                            eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
                            for generated_ar in generated_ar_batch:
                                if eos_a_id not in generated_ar:
                                    logging.info(self.tokenizer.decode(generated_ar))
                            batch_size = len(turn_batch['dial_id'])
                            decoded_batch = {'resp': [[] for _ in range(batch_size)], 'bspn': [[] for _ in range(batch_size)], 'aspn': [[] for _ in range(batch_size)]}
                        #---------------------------------------------------------------
                        

                        # 2022-05-11
                        # try:
                        #     decoded = self.decode_generated_act_resp(generated)
                        # except ValueError as exception:
                        #     logging.info(str(exception))
                        #     logging.info(self.tokenizer.decode(generated))
                        #     decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else: # predict bspn, access db, then generate act and resp
                        ##### 2022/1/13 Using 'try' 'except' to handle gpt2's max length#####
                        try:
                            outputs_batch = self.model.generate(input_ids=inputs['context_tensor'], attention_mask=attention_mask,
                                                        max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                                        pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0],
                                                        use_cache=cfg.use_cache)
                        except RuntimeError:
                            logging.info("\'bspan\' is too long, max sentence length of GPT-2 is 1024.(batch iondex:{}, turn number:{})".format(batch_idx, turn_num))
                        #####################
                        generated_bs_batch = outputs_batch.cpu().numpy()
                        bspn_gen_batch = self.decode_generated_bspn_batch(generated_bs_batch[:, context_length-1:]) # 已测试 2021/12/31
                        # check DB result
                        if cfg.use_true_db_pointer: # 该分支未测试 2021/12/31
                            db_batch = []
                            for db in turn_batch['db']:
                                db_batch.append(db)
                        else:
                            turn_domain_batch = []
                            for turn_domain in turn_batch['turn_domain']:
                                turn_domain_batch.append(turn_domain)   

                            bspan_batch = []
                            for bspn_gen in bspn_gen_batch:
                                bspan_batch.append(self.tokenizer.decode(bspn_gen))     

                            db_result_batch = self.reader.bspan_to_DBpointer_batch(bspan_batch, turn_domain_batch)
                            db_batch = []
                            for db_result in db_result_batch:
                                db_batch.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')))
                        # Padding context list, and convert to tensor
                        inputs['context_db'] = []
                        for context, bspn_gen, db in zip(inputs['context'], bspn_gen_batch, db_batch):
                            inputs['context_db'].append(context[:-1] + bspn_gen + db + self.tokenizer.encode(['<sos_a>']))
                         
                        inputs, _, attention_mask = self.add_torch_input_eval_batch(inputs, context_name='context_db')
                        context_length = inputs['context_db_tensor'].shape[1]
                        ##### 2021/1/13 #####
                        try:
                            outputs_db = self.model.generate(input_ids=inputs['context_db_tensor'], attention_mask=attention_mask,
                                                        max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                                        pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0],
                                                        use_cache=cfg.use_cache)
                        except RuntimeError:
                            logging.info("\'aspan\' and \'rspan\' is too long, max sentence length of GPT-2 is 1024.(batch iondex:{}, turn number:{})".format(batch_idx, turn_num))
                        #####################
                        generated_ar_batch = outputs_db.cpu().numpy()
                        generated_ar_batch = generated_ar_batch[:, context_length-1:].tolist()
                        try:
                            decoded_batch = self.decode_generated_act_resp_batch(generated_ar_batch, turn_batch['dial_id'], turn_num)
                            decoded_batch['bspn'] = bspn_gen_batch
                        except ValueError as exception:
                            logging.info(str(exception))
                            # logging
                            eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
                            for generated_ar in generated_ar_batch:
                                if eos_a_id not in generated_ar:
                                    logging.info(self.tokenizer.decode(generated_ar))
                            batch_size = len(turn_batch['dial_id'])
                            decoded_batch = {'resp': [[] for _ in range(batch_size)], 'bspn': [[] for _ in range(batch_size)], 'aspn': [[] for _ in range(batch_size)]}
                    
                    
                    turn_batch['resp_gen'] = decoded_batch['resp']
                    turn_batch['bspn_gen'] = turn_batch['bspn'] if cfg.use_true_curr_bspn else decoded_batch['bspn']
                    turn_batch['aspn_gen'] = turn_batch['aspn'] if cfg.use_true_curr_aspn else decoded_batch['aspn']
                    turn_batch['dspn_gen'] = turn_batch['dspn']
                    
                    pv_turn_batch['labels'] = inputs['labels'] # all true previous context
                    pv_turn_batch['resp'] = turn_batch['resp'] if cfg.use_true_prev_resp else decoded_batch['resp']
                    pv_turn_batch['bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn else decoded_batch['bspn']
                    pv_turn_batch['db'] = turn_batch['db'] if cfg.use_true_curr_bspn else db_batch
                    pv_turn_batch['aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded_batch['aspn']

                    # state = self.reader.bspan_to_constraint_dict(bspn)
                    

                result_collection.update(
                    self.reader.inverse_transpose_batch(dial_batch))

        logging.info("Inference time: {:.2f} min".format((time.time()-btm)/60))
        if cfg.multi_gpu: # 'all_gather_object' way
            output_list = [None for _ in range(cfg.world_size)]
            output_list = self.all_gather_object(obj=result_collection, object_list=output_list)
            if cfg.local_rank == 0:
                result_collection = {dial_id : dial_content for output in output_list for dial_id, dial_content in output.items()}
            else:
                return # finish child process

        # Compute score
        if (cfg.multi_gpu and cfg.local_rank == 0) or (not cfg.multi_gpu):
            # score
            btm = time.time()
            results, _ = self.reader.wrap_result_lm(result_collection)
            # bleu, success, match = self.evaluator.validation_metric(results)
            ##### 2022/1/6 add success, inform #####
            bleu, success, match, results = self.evaluator.validation_metric_results(results)
            ####################
            logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
            score = 0.5 * (success + match) + bleu
            valid_loss = 130 - score
            logging.info('Validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
                match, success, bleu, score))
            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)

            
            model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
            eval_on = '-'.join(cfg.exp_domains)
            if data == 'test':
                eval_on += '_test'
            if not os.path.exists(cfg.log_path):
                os.mkdir(cfg.log_path)
            log_file_name = os.path.join(cfg.log_path, model_setting+'-'+eval_on+'.json')
            if os.path.exists(log_file_name):
                eval_to_json = json.load(open(log_file_name, 'r'))
                eval_to_json[epoch_setting] = eval_results
                json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
            else:
                eval_to_json = {}
                eval_to_json[epoch_setting] = eval_results
                json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
            logging.info('Update eval results to {}'.format(log_file_name))

            # save results in json 2022/1/5
            # eval_load_path : 'experiments/all_best_model_sd11_lr0.0001_bs2_ga16_ctUBARU_GEN_CONTEXT/epoch1_trloss0.07_gpt2'
            if data == 'test':
                inference_result_file_name = os.path.join(cfg.eval_load_path,  'inference-test.json')
            else:
                inference_result_file_name = os.path.join(cfg.eval_load_path, 'inference-validate.json')
            with open(inference_result_file_name, 'w') as result_writer:
                json.dump(results, result_writer, indent=2) 
            logging.info('Save json results to {}'.format(inference_result_file_name))

            # state是GT state("GT state"指的是mwz2.0数据集中标注的state)
            save_standard_results_gt_state = True
            if save_standard_results_gt_state:
                if data == 'test':
                    inference_result_file_name = os.path.join(cfg.eval_load_path,  'inference-test-standard-gt-state.json')
                else:
                    inference_result_file_name = os.path.join(cfg.eval_load_path, 'inference-validate-standard-gt-state.json')
                with open(inference_result_file_name, 'w') as result_writer:
                    standard_results = self._change_dataformat_to_standard(results, 'GT')
                    json.dump(standard_results, result_writer, indent=2) 
                logging.info('Save standard GT state json results to {}'.format(inference_result_file_name))

            # state是gen state（"gen state"指的是模型生成的state）
            save_standard_results_gen_state = True
            if save_standard_results_gen_state:
                if data == 'test':
                    inference_result_file_name = os.path.join(cfg.eval_load_path,  'inference-test-standard-gen-state.json')
                else:
                    inference_result_file_name = os.path.join(cfg.eval_load_path, 'inference-validate-standard-gen-state.json')
                with open(inference_result_file_name, 'w') as result_writer:
                    standard_results = self._change_dataformat_to_standard(results, 'GEN')
                    json.dump(standard_results, result_writer, indent=2) 
                logging.info('Save standard generated state json results to {}'.format(inference_result_file_name))
            
            return eval_results

######################################### _change_dataformat_to_standard (2022/3/10) ################
    def _change_dataformat_to_standard(self, ubar_res_dict, state_type):
        standard_res_dict = {}
        dial_id, cur_standard_dial_list, cur_standard_turn_dict = None, None, None
        
        for dial in ubar_res_dict:
            cur_standard_dial_list = []
            dial_id = dial[0]['dial_id']

            for turn in dial[1:]:
                cur_standard_turn_dict = {}

                # Get generated response
                cur_standard_turn_dict['response'] = turn['resp_gen']

                # Get state('GEN' or 'GT')
                if state_type == 'GT':
                    cur_state = self.reader.bspan_to_constraint_dict(turn['bspn'])
                if state_type == 'GEN':
                    cur_state = self.reader.bspan_to_constraint_dict(turn['bspn_gen'])

                cur_standard_turn_dict['state'] = cur_state
            
                cur_standard_dial_list.append(cur_standard_turn_dict)

            standard_res_dict[dial_id] = cur_standard_dial_list
        
        return standard_res_dict
#####################################################################################################

######################################### Realize 'all_gather' (2022/1/4) ###########################
    
    def _object_to_tensor(self, obj):
        f = io.BytesIO() 
        self._pickler(f).dump(obj)
        byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage)
        local_size = torch.LongTensor([byte_tensor.numel()])
        return byte_tensor, local_size


    def _tensor_to_object(self, tensor, tensor_size):
        buf = tensor.numpy().tobytes()[:tensor_size]
        return self._unpickler(io.BytesIO(buf)).load()
    

    def all_gather_object(self, obj, object_list):
        group_size = dist.get_world_size()
        object_list = [None for _ in range(group_size)]

        # Conver 'Object' to 'Tensor'
        object_tensor, local_size = self._object_to_tensor(obj)

        # Move 'object_tensor' and 'local_size' to gpu
        input_tensor = object_tensor.to(self.device)
        local_size = local_size.to(self.device)
        object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=self.device) # object_sizes_tensor shape: [group_size, ] 
        object_size_list = [
                object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
            ]
        
        # first, gather local_size
        dist.all_gather(object_size_list, local_size)
        max_object_size = int(max(object_size_list).item())
        # Resize tensor to max size across all ranks.
        input_tensor.resize_(max_object_size)
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=self.device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
        dist.all_gather(output_tensors, input_tensor)
        for i, tensor in enumerate(output_tensors):
            tensor = tensor.type(torch.uint8)
            if tensor.device != torch.device("cpu"):
                tensor = tensor.cpu()
            tensor_size = object_size_list[i]
            object_list[i] = self._tensor_to_object(tensor, tensor_size)
        return object_list

##########################################################################################


    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated)-1
            logging.info('\'<eos_r>\' not in generated: ' + self.tokenizer.decode(generated))
        # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1
        
        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx+1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx+1]
            decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        # if cfg.use_true_curr_bspn:
            
        # else:  # predict bspn aspn resp
        #     eos_b_idx = generated.index(eos_b_id)
        #     eos_a_idx = generated.index(eos_a_id)
        #     decoded['bspn'] = generated[: eos_b_idx+1]
        #     decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        #     decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded
    
    ##### 2021/12/31 #####
    def decode_generated_act_resp_batch(self, generated_batch, dial_id_batch, turn_num):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded_batch = {}
        decoded_batch['aspn'] = []
        decoded_batch['resp'] = []

        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        for generated, dial_id in zip(generated_batch, dial_id_batch):
            # eos_r may not exists if gpt2 generated repetitive words.
            if eos_r_id in generated:
                eos_r_idx = generated.index(eos_r_id)
            else:
                eos_r_idx = len(generated)-1
                logging.info('\'<eos_r>\' not in generated(dialog id {}, turn num {}): '.format(dial_id, turn_num) + self.tokenizer.decode(generated))
            
            if cfg.use_true_curr_aspn:  # only predict resp
                decoded_batch['resp'].append(generated[: eos_r_idx+1])
            else:  # predicted aspn, resp
                eos_a_idx = generated.index(eos_a_id)
                decoded_batch['aspn'].append(generated[: eos_a_idx+1])
                decoded_batch['resp'].append(generated[eos_a_idx+1: eos_r_idx+1])
            
        return decoded_batch
    ######################

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated)-1
        return generated[: eos_b_idx+1]

    ######################## 2021/12/29 ###################
    def decode_generated_bspn_batch(self, generated_batch):
        bspn_gen_batch = []
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        for generated in generated_batch:
            # from numpy to list
            generated = generated.tolist()

            if eos_b_id in generated:
                eos_b_idx = generated.index(eos_b_id)
            else:
                eos_b_idx = len(generated)-1
            bspn_gen_batch.append(generated[: eos_b_idx+1])
        return bspn_gen_batch
    ########################################################

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    # add '--local_rank' (2021/12/17)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        # cfg.model_path = cfg.eval_load_path
        cfg.gpt_path = cfg.eval_load_path
    else:  # train
        parse_arg_cfg(args)

        ##### Load trained model for generated context UBARU 2021/12/29 ##### --> 是否使用预训练模型
        # if cfg.context_scheme == 'UBARU_GEN_CONTEXT':
            # cfg.gpt_path = cfg.eval_load_path
            # logging.info("Load trained model from {} for generated context UBARU".format(cfg.gpt_path))
        #####################################################################

        if cfg.exp_path in ['', 'to be generated']:
            # log file path, control the factors: seed, learning_rate, batch_size, early_stop_count, weight decay...
            # cfg.exp_path = 'experiments/{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}/'.format('-'.join(cfg.exp_domains),
            #                                                                     cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
            #                                                                     cfg.early_stop_count, cfg.weight_decay_count)
            
            experiments_path = './experiments' if 'all' in cfg.exp_domains else './experiments_Xdomain'
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_{}_sd{}_lr{}_bs{}_ga{}_en{}_ws{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.context_scheme, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps, cfg.epoch_num, cfg.warmup_steps))
            ##### 2022/1/9 #####
            if cfg.context_scheme == 'UBARU_GEN_CONTEXT':
                cfg.exp_path += '_{}'.format(cfg.gen_context)
            ####################
            logging.info('Save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if cfg.multi_gpu == False:
            # torch.cuda.set_device(cfg.cuda_device[0])
            device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
        else:
            # Init process group(2021/12/17)
            dist.init_process_group(backend='nccl')
            cfg.local_rank = dist.get_rank()
            cfg.world_size = dist.get_world_size()
            device = torch.device("cuda:{}".format(cfg.cuda_device[cfg.local_rank]))    
            
    else:
        device = torch.device('cpu')
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)

    if args.mode == 'train':    # train
        if cfg.save_log:  # save cfg details.
            pass
        if cfg.context_scheme == 'UBARU':
            m.train()
        elif cfg.context_scheme == 'URURU':
            m.train_URURU()
        elif cfg.context_scheme == 'UBARU_GEN_CONTEXT':
            m.train_UBARU_GEN_CONTEXT()
        else:
            logging.info('Invalid context Scheme. must be UBARU or URURU')
            exit()
    elif args.mode == 'adjuest':
        pass
    else:  # test
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                        ))

        if cfg.context_scheme == 'UBARU':
            ##### 2021/12/28 #####
            # m.validate_batch()
            m.validate_batch('test') 
            ######################
            # m.validate()
            # m.validate('test')
        elif cfg.context_scheme == 'URURU':
            m.validate_URURU()
            m.validate_URURU('test')
        elif cfg.context_scheme == 'UBARU_GEN_CONTEXT':
            m.validate_UBARU_GEN_CONTEXT()
            m.validate_UBARU_GEN_CONTEXT('test')

        # logging.info('Running eavl on test')
        # m.validate('test')


#  testing:  python train.py -mode test -cfg eval_load_path=experiments/all__sd11_lr0.001_bs2_ga8/epoch5_trloss0.80_gpt2/


if __name__ == "__main__":
    main()
