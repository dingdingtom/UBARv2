from init import *

from load_data import *
from init_logger import *
from reader import MultiWozReader
from config import config_global as cfg

#----------------------------------------
# UBAR
class ModelUbar2(object):
    def __init__(self, args, logger):
        
        self.args = args
        self.logger = logger
        
        device = args.device
        dir_pack = args.dir_pack
        
        mode = cfg.mode
        dir_gpt = cfg.dir_gpt
        
        if dir_pack != '':
            dir_gpt = dir_pack
        
        #--------------------
        tokenizer = GPT2Tokenizer.from_pretrained(dir_gpt)
        reader = MultiWozReader(tokenizer)
        data = reader.data
        model = GPT2LMHeadModel.from_pretrained(dir_gpt)
        model = model.to(device)
        if mode == 'train':
            model.resize_token_embeddings(len(tokenizer))
            logInfo(logger, f'init: resize_token_embeddings')
        
        logInfo(logger, f'init: tokenizer, reader, model')
        logInfo(logger, f'dir gpt: {dir_gpt}')

        cfg.pad_id = tokenizer.encode('<pad>')[0]
        
        #--------------------
        self.data = data
        self.model = model
        self.reader = reader
        self.tokenizer = tokenizer
        
        return
    
    #----------------------------------------
    def getOptimizerScheduler(self, tot_batch_train_node):
        
        args = self.args
        model = self.model
        
        accumulation = args.accumulation
        lr = args.lr
        num_step_warmup = args.num_step_warmup
        type_optimizer = args.type_optimizer
        type_scheduler = args.type_scheduler
        weight_decay = args.weight_decay
        
        #--------------------
        # optimizer
        if type_optimizer == 'adamw':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            
        #--------------------
        # scheduler
        tot_batch_optim_node = tot_batch_train_node // accumulation
        tot_batch_warmup_node = num_step_warmup
        if not tot_batch_warmup_node > 0:
            tot_batch_warmup_node = int(tot_batch_optim_node * 0.2)
            
        dic_type_scheduler = {
            'constw': optimization.get_constant_schedule_with_warmup,
            'cosw': optimization.get_cosine_schedule_with_warmup,
            'linw': optimization.get_linear_schedule_with_warmup,
            'polyw': optimization.get_polynomial_decay_schedule_with_warmup
        }
        
        dic_args_scheduler = {
            'optimizer': optimizer,
            'num_warmup_steps': tot_batch_warmup_node,
        }
        if 'const' not in type_scheduler:
            dic_args_scheduler['num_training_steps'] = tot_batch_optim_node
        scheduler = dic_type_scheduler[type_scheduler](**dic_args_scheduler)
        
        return optimizer, scheduler, tot_batch_optim_node, tot_batch_warmup_node
    
    #----------------------------------------
    def forward(self, inputs, labels, masks):
        
        # inputs: Tensor | shape (batch_size, maxlen) | 输入文本
        # labels: Tensor | shape (batch_size, maxlen) | 输入文本
        
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        
        device = args.device
        level_cuda = args.level_cuda
        lst_name_element_mask_input1 = args.lst_name_element_mask_input1
        lst_name_element_mask_input2 = args.lst_name_element_mask_input2
        lst_name_element_mask_label = args.lst_name_element_mask_label
        local_rank = args.local_rank
        p_mask_input1 = args.p_mask_input1
        p_mask_input2 = args.p_mask_input2
        p_mask_label = args.p_mask_label
        rate_kl = args.rate_kl
        target_mask = args.target_mask
        use_amp = args.use_amp
        use_gen_mask = args.use_gen_mask
        use_same_mask = args.use_same_mask
        
        dic_element_name2info = cfg.dic_element_name2info
        lst_name_element = cfg.lst_name_element
        pad_id = cfg.pad_id
        
        #----------
        # 计算 Cross Entropy 之前需要 mask 不需要的部分
        # 需要 mask 的位置：夹在指定元素的 sos 和 eos 之间
        inputs1 = torch.clone(inputs)
        inputs2 = torch.clone(inputs)
        labels_shift = labels[:, 1:].reshape(-1)
        # labels_shift: Tensor | shape (batch_size * (maxlen - 1),) | 各个文本除了首位的下标

        if p_mask_input1 > 0 or p_mask_input2 > 0 or p_mask_label > 0:
            ten_mask_all_se_input1 = torch.zeros_like(inputs, dtype=torch.bool)
            ten_mask_all_se_input2 = torch.zeros_like(inputs, dtype=torch.bool)
            ten_mask_all_se_label = torch.zeros_like(inputs, dtype=torch.bool)
            for name_element in lst_name_element:
                # 该部分的信息
                str_sos = dic_element_name2info[name_element]['sos']
                str_eos = dic_element_name2info[name_element]['eos']
                # str_sos: Str | 该部分的开头标志词语串 | ex. '<sos_b>'
                id_sos = tokenizer.encode(str_sos)[0]
                id_eos = tokenizer.encode(str_eos)[0]
                # id_eos: int | 生成时用于判断结束 | ex. 50308
                # 用或操作更新需要 mask 的位置
                if name_element in lst_name_element_mask_input1:
                    ten_mask_all_se_input1 += (inputs == id_sos) + (inputs == id_eos)
                if name_element in lst_name_element_mask_input2:
                    ten_mask_all_se_input2 += (inputs == id_sos) + (inputs == id_eos)
                if name_element in lst_name_element_mask_label:
                    ten_mask_all_se_label += (inputs == id_sos) + (inputs == id_eos)
                    
            
            def getMask(ten_mask_all_se, p_mask):

                ten_mask_all_se_cumsum = ten_mask_all_se.cumsum(dim=-1)
                mat_mask_all_se_cumsum = ten_mask_all_se_cumsum.cpu().numpy()
                mat_mask_all_se_cumsum_rdiff = np.diff(
                    mat_mask_all_se_cumsum[:, ::-1], 
                    append=mat_mask_all_se_cumsum[:, [0]] - 1
                )[:, ::-1]
                mat_mask_all_flag1 = mat_mask_all_se_cumsum % 2 == 1
                mat_mask_all_flag2 = mat_mask_all_se_cumsum_rdiff == 0
                mat_mask_all = np.bitwise_and(mat_mask_all_flag1, mat_mask_all_flag2)
                
                mat_mask_all_flag3 = np.random.rand(*mat_mask_all.shape) < p_mask
                mat_mask_all = np.bitwise_and(mat_mask_all, mat_mask_all_flag3)
                ten_mask_all = torch.from_numpy(mat_mask_all).to(device)
                
                # inputs: Tensor | shape (batch_size, maxlen) | 输入文本
                #     ex. [[50312, 201, 50306, 50314, 202, 202, 50308]]
                #     ex. <sos_u> a <eos_u> <sos_b> b b <eos_b> （需要 mask 的是 user 的内容）
                # ten_mask_all_se: Tensor | shape (batch_size, maxlen) | 布尔标记所有 sos 和 eos
                #     ex. [[True, False, True, False, False, False, False]]
                # mat_mask_all_se_cumsum: array | shape (batch_size, maxlen) | 累计标记数
                #     ex. [[1, 1, 2, 2, 2, 2, 2]]
                # mat_mask_all_se_cumsum_rdiff: array | shape (batch_size, maxlen) | 差分倒序，结果最前补个 -1
                #     ex. [[-1, 0, -1, 0, 0, 0, 0]]
                # mat_mask_all_flag1: array | shape (batch_size, maxlen) | 需要 mask 的条件1：累计标记数为奇数
                #     ex. [[True, True, False, False, False, False, False]]
                # mat_mask_all_flag2: array | shape (batch_size, maxlen) | 需要 mask 的条件2：差分倒序为 0 
                #     ex. [[False, True, False, True, True, True, True]]
                # mat_mask_all: array | shape (batch_size, maxlen) | 需要 mask 的位置：夹在指定元素的 sos 和 eos 之间
                #     ex. [[False, True, False, False, False, False, False]]
                
                return ten_mask_all
            
            if 'input' in target_mask:
                ten_mask_all_input1 = getMask(ten_mask_all_se_input1, p_mask_input1)
                ten_mask_all_input2 = getMask(ten_mask_all_se_input2, p_mask_input2)     
                if use_gen_mask:
                    ten_mask_all_input1 = torch.bitwise_and(ten_mask_all_input1, masks)
                    ten_mask_all_input2 = torch.bitwise_and(ten_mask_all_input2, masks)
                else:
                    ten_mask_all_input1 = torch.bitwise_and(ten_mask_all_input1, torch.bitwise_not(masks))
                    ten_mask_all_input2 = torch.bitwise_and(ten_mask_all_input2, torch.bitwise_not(masks))
                inputs2[ten_mask_all_input2] = pad_id
                
                if use_same_mask:
                    inputs1[ten_mask_all_input2] = pad_id
                else:
                    inputs1[ten_mask_all_input1] = pad_id
            if 'label' in target_mask:
                ten_mask_all_label = getMask(ten_mask_all_se_label, p_mask_label)
                ten_mask_all_label_shift = ten_mask_all_label[:, 1:].reshape(-1)
                # ten_mask_all_shift: Tensor | shape (batch_size * (maxlen - 1),) | 各个文本除了首位的 mask
                labels_shift[ten_mask_all_label_shift] = pad_id 
        
        #----------
        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        if level_cuda == 2:
            criterion = criterion.to(local_rank)
        
        #----------
        # 前向过程
        if use_amp:
            context_forward = autocast
        else:
            context_forward = nullcontext
        
        if rate_kl:
            with context_forward():
                logits1 = model(inputs1)[0]
                logits2 = model(inputs2)[0]
                num_vocab = logits1.shape[-1]
                logits1_shift = logits1[:, :-1, :].reshape(-1, num_vocab)
                logits2_shift = logits2[:, :-1, :].reshape(-1, num_vocab)
                # logits: Tensor | shape (batch_size, maxlen, num_vocab) | 由输入文本得到的每个位置分数
                # logits_shift: Tensor | shape (batch_size * (maxlen - 1), num_vocab) | 各个文本除了末位的分数
                
                #----------
                # 计算 mask 后的损失
                # Cross Entropy
                loss1_ce_sum = criterion(logits1_shift, labels_shift)
                loss2_ce_sum = criterion(logits2_shift, labels_shift)
                loss_ce_sum = (loss1_ce_sum + loss2_ce_sum) / 2
                num_word = int((labels_shift != pad_id).sum())
                loss_ce_sum = loss_ce_sum / num_word
                # Kullback-Leibler Divergence
                logits1_shift = logits1_shift[labels_shift != pad_id]
                logits2_shift = logits2_shift[labels_shift != pad_id]
                loss1_kl_sum = F.kl_div(
                    F.log_softmax(logits2_shift, dim=-1), 
                    F.softmax(logits1_shift, dim=-1),
                    reduction='sum'
                )
                loss2_kl_sum = F.kl_div(
                    F.log_softmax(logits1_shift, dim=-1), 
                    F.softmax(logits2_shift, dim=-1),
                    reduction='sum'
                )
                loss_kl_sum = (loss1_kl_sum + loss2_kl_sum) / 2
                # 汇总
                loss_sum = loss_ce_sum + rate_kl * loss_kl_sum
    
                loss = loss_sum
                loss_ce = loss_ce_sum
                loss_kl = loss_kl_sum
        else:
            with context_forward():
                logits = model(inputs1)[0]
                num_vocab = logits.shape[-1]
                logits_shift = logits[..., :-1, :].reshape(-1, num_vocab)
                loss_sum = criterion(logits_shift, labels_shift)
                # 以句子长度求 LM 损失的加权平均
                num_word = int((labels_shift != pad_id).sum())
                loss = loss_sum / num_word
                # loss: batch 中每个句子的 LM 损失（已对长度、batch_size平均）
                loss_ce = loss
                loss_kl = loss
        
        return loss, loss_ce, loss_kl
    
    #----------------------------------------
    def train(self):

        args = self.args
        logger = self.logger
        model = self.model
        reader = self.reader
        tokenizer = self.tokenizer
        
        accumulation = args.accumulation
        batch_size_train = args.batch_size_train
        dependency = args.dependency
        device = args.device
        dic_element_name2coef_train = args.dic_element_name2coef_train
        flag_master = args.flag_master
        level_cuda = args.level_cuda
        max_epoch = args.max_epoch
        max_grad_norm = args.max_grad_norm
        num_beams_train = args.num_beams_train
        p_gen_lower = args.p_gen_lower
        p_gen_upper = args.p_gen_upper
        top_p_train = args.top_p_train
        use_amp = args.use_amp
        use_raw_cat = args.use_raw_cat
        use_raw_context = args.use_raw_context
        
        dic_element_name2info = cfg.dic_element_name2info
        dir_exp = cfg.dir_exp
        lst_name_element = cfg.lst_name_element
        maxlen = cfg.maxlen
        mode = cfg.mode
        num_low_resource = args.num_low_resource
        pad_id = cfg.pad_id
        use_low_resource = args.use_low_resource
        
        #--------------------
        # 命令行参数
        lst_str_arg = []
        str_arg_all = 'args: \n'
        for k, v in args._get_kwargs():
            if type(v) in [str, int, float]:
                lst_str_arg.append(f'{k:<22}: {v:<8}')
        for i_arg, str_arg in enumerate(lst_str_arg, 1):
            str_arg_all += str_arg
            str_arg_all += '\n' if i_arg % 3 == 0 else ' | '
        str_arg_all += '\n'
        logInfo(logger, f'{str_arg_all}')
        
        #--------------------
        # 训练：数据集、加载器、优化器
        # 验证：数据集、加载器
        dic_name2dials = {'train': reader.train, 'test': reader.test, 'val': reader.dev}
        lst_dial_train = dic_name2dials['train']
        
        if mode.startswith('train'):
            if use_low_resource:
                lst_dial_train = random.sample(lst_dial_train, num_low_resource)
                logInfo(logger, f'low resource train: {num_low_resource} dialogs')

            if mode == 'train':
                dataset_train = DialogDatasetSessionLevel(lst_dial_train)
            elif mode == 'train2':
                dataset_train = DialogDatasetTurnLevel(lst_dial_train)
            dataset_train.getListBatch(batch_size=batch_size_train)

            #--------------------
            # 数据加载器
            if level_cuda == 2:
                shuffle = False
                sampler_train = DistributedSampler(dataset_train)
            else:
                shuffle = True
                sampler_train = None
            loader_train = DataLoader(dataset_train, batch_size=1, shuffle=shuffle,
                sampler=sampler_train, collate_fn=collate_fn)

            #--------------------
            # 数据集信息
            tot_session_train = dataset_train.tot_session
            tot_turn_train = dataset_train.tot_turn
            tot_batch_train = int(tot_session_train / batch_size_train * max_epoch)
            if level_cuda == 2:
                tot_batch_train_node = len(loader_train) * max_epoch
            else:
                tot_batch_train_node = tot_batch_train

            logInfo(logger, f'total session train: {tot_session_train}')
            logInfo(logger, f'total turn train: {tot_turn_train}')
            logInfo(logger, f'total batch train: {tot_batch_train}')
            logInfo(logger, f'total batch train node: {tot_batch_train_node}')

            #--------------------
            # 优化器
            optimizer, scheduler, tot_batch_optim_node, tot_batch_warmup_node = \
                self.getOptimizerScheduler(tot_batch_train_node)
            logInfo(logger, f'init: optimizer, scheduler')
            logInfo(logger, f'total batch optim node: {tot_batch_optim_node}')
            logInfo(logger, f'total batch warmup node: {tot_batch_warmup_node}')

            # 混合精度梯度缩放
            if use_amp:
                scaler = GradScaler()
                logInfo(logger, f'init: amp-scaler')
        
        #--------------------
        # 中间统计变量
        epoch_start = 1
        cnt_batch_train_node = 0
        lst_loss_avg = []
        
        flag_epoch_train = (mode == 'train2')

        #----------
        # 初始化进度条
        if flag_master:
            pbar = tqdm(total=max_epoch)
            
        #----------
        # 每轮训练
        for i_epoch in range(epoch_start, max_epoch + 1):
            time.sleep(0.05)
            time_cur = time.time()
        
            #----------
            # epoch train
            if flag_epoch_train:
                model.train()
                model_train2 = model
                if level_cuda == 2:
                    loader_train.sampler.set_epoch(i_epoch)
                    model_train2 = model.module
                loss_sum = 0
                loss_ce_sum = 0
                loss_kl_sum = 0
                num_sample_sum = 0
                num_turn_sum = 0
                cnt_gen_turn = 0
                for i_batch, batch_dial in enumerate(loader_train, 1):
                    # batch_dial: List[List[Dict]] | 
                    #     每个元素是一个对话，对话是列表，它的每个元素是一轮，轮是字典
                    batch_size = len(batch_dial)
                    num_turn = len(batch_dial[0])
                    lst_context_raw = [[] for i in range(batch_size)]
                    lst_context_replace = [[] for i in range(batch_size)]
                    lst_flag_gen = [[] for i in range(batch_size)]
                    
                    # 各个对话逐轮生成
                    for i_turn in range(num_turn):
                        lst_turn_cur = [dial[i_turn] for dial in batch_dial]
                        flag_gen_element = False
                        # 每个对话的该轮，逐个生成设定为 Generated 的部分
                        lst_cat_raw = [[] for i in range(batch_size)]
                        lst_cat_replace = [[] for i in range(batch_size)]
                        for name_element in lst_name_element:
                            # 该部分的信息
                            str_sos = dic_element_name2info[name_element]['sos']
                            str_eos = dic_element_name2info[name_element]['eos']
                            lst_start = tokenizer.encode(str_sos)
                            lst_end = tokenizer.encode(str_eos)
                            # str_sos: Str | 该部分的开头标志词语串 | ex. '<sos_b>'
                            # lst_start: List[int] | 该词语下标独自组成的列表 | ex. [50314]
                            id_pad = tokenizer.eos_token_id
                            id_sos = lst_start[0]
                            id_eos = lst_end[0]
                            # id_pad： int | 生成时用于填充 | ex. 50256
                            # id_eos: int | 生成时用于判断结束 | ex. 50308
                            coef_p_gen = dic_element_name2coef_train.get(name_element, 0)
                            p_gen = p_gen_lower + coef_p_gen * (p_gen_upper - p_gen_lower) * (i_epoch / max_epoch)
                            flag_gen_element = self.updateFlagGenElement(name_element, p_gen, dependency, flag_gen_element)
                            # flag_gen_element: bool | 各个对话在该轮的该部分，都要用生成的来更新 context
                            
                            # 该部分的开头标志词语的下标，新增到 context 中
                            for i in range(batch_size):       
                                lst_cat_raw[i] += lst_start
                                lst_cat_replace[i] += lst_start
                                lst_flag_gen[i] += [False]

                            # 有的部分需要生成
                            if name_element in ['user', 'db']:
                                pass
                            elif flag_gen_element:
                                # 用预训练语言模型的 `generate` 生成该部分，并在原 context 之后取出实际生成的部分
                                #     用 `tokenizer.eos_token_id` 作为填充
                                #     该部分词语的结尾下标作为判断停止生成标志
                                #     每个对话的 context 长度不一样，需要在左边填充
                                if use_raw_context and use_raw_cat:
                                    lst_context_cur = [lst_context_raw[i] + lst_cat_raw[i] for i in range(batch_size)]
                                elif use_raw_context and not use_raw_cat:
                                    lst_context_cur = [lst_context_raw[i] + lst_cat_replace[i] for i in range(batch_size)]
                                elif not use_raw_context and use_raw_cat:
                                    lst_context_cur = [lst_context_replace[i] + lst_cat_raw[i] for i in range(batch_size)]
                                else:
                                    lst_context_cur = [lst_context_replace[i] + lst_cat_replace[i] for i in range(batch_size)]
                                lst_context_cur =[lst_context_cur[i][-(maxlen-128):] for i in range(batch_size)]

                                maxlen_context, lst_context_pad = self.padList(lst_context_cur, id_pad)
                                ten_context_pad = LongTensor(lst_context_pad).to(device)
                                # maxlen_context: int | 各个对话的最长 context 的长度
                                # lst_context_pad: List[List[int]] | 在左边填充后的各个对话的 context 
                                # ten_context_pad: Tensor | 上述列表转为张量 | shape: (batch_size, maxlen_context)
                                ten_sents = model_train2.generate(
                                    input_ids=ten_context_pad,
                                    max_length=maxlen,
                                    temperature=0.7,
                                    num_beams=num_beams_train,
                                    top_p=top_p_train,
                                    pad_token_id=id_pad, 
                                    eos_token_id=id_eos
                                )
                                ten_gen = ten_sents[:, maxlen_context:]
                                lst2_gen = ten_gen.cpu().numpy().tolist()
                                # ten_sents: Tensor | 各个对话的 context + gen，在右边填充 | shape: (batch_size, maxlen_sent)
                                # ten_gen: Tensor | 各个对话的 gen，在右边填充 | shape: (batch_size, maxlen_gen)
                                # lst2_gen: List[List[int]] | 上述张量转为列表 | shape: (batch_size, maxlen_gen)

                            # 更新原来的各个对话的 context，按概率用 Ground Truth 或 Generated，在后面截断以留位置给下一轮生成
                            for i in range(batch_size):
                                # 更新原始部分的拼接
                                lst_cat_raw[i] += lst_turn_cur[i][name_element][1:-1]
                                lst_cat_raw[i] += lst_end
                                # 更新替换部分的拼接
                                if name_element in ['user', 'db'] or not flag_gen_element:
                                    lst_cat_replace[i] += lst_turn_cur[i][name_element][1:-1]
                                    lst_flag_gen[i] += [False] * len(lst_turn_cur[i][name_element][1:-1])
                                elif flag_gen_element:
                                    lst_gen = lst2_gen[i]
                                    # 各个对话的 gen 最后应为结尾标志词语的下标
                                    ind_gen_end = len(lst_gen) - 1
                                    while ind_gen_end and lst_gen[ind_gen_end] in [id_pad, id_eos]:
                                        ind_gen_end -= 1
                                    lst_cat_replace[i] += lst_gen[:ind_gen_end+1]
                                    lst_flag_gen[i] += [True] * len(lst_gen[:ind_gen_end+1])
                                lst_cat_replace[i] += lst_end
                                lst_flag_gen[i] += [False]
        
                        # 一轮结束，更新最终上下文 
                        for i in range(batch_size):
                            lst_context_raw[i] += lst_cat_raw[i]
                            lst_context_raw[i] = lst_context_raw[i][-(maxlen-128):]
                            lst_context_replace[i] += lst_cat_replace[i]
                            lst_context_replace[i] = lst_context_replace[i][-(maxlen-128):]
                            lst_flag_gen[i] = lst_flag_gen[i][-(maxlen-128):]
                        cnt_gen_turn += flag_gen_element
                    
                    # 所有轮结束，计算损失
                    maxlen_context, lst_context_pad = self.padList(lst_context_replace, pad_id, mode='right')
                    maxlen_flag_gen, lst_flag_gen_pad = self.padList(lst_flag_gen, pad_id, mode='right')
                    batch_dial = LongTensor(lst_context_pad).to(device)
                    batch_mask = LongTensor(lst_flag_gen_pad).bool().to(device)
                    # batch_dial: shape (batch_size, maxlen_batch)
                    loss, loss_ce, loss_kl = self.forward(inputs=batch_dial, labels=batch_dial, masks=batch_mask)
                    num_sample = len(batch_dial)

                    #----------
                    # 此 batch 更新 epoch 统计信息
                    cnt_batch_train_node += 1
                    num_sample_sum += num_sample
                    loss_sum += loss.item() * num_sample
                    loss_ce_sum += loss_ce.item() * num_sample
                    loss_kl_sum += loss_kl.item() * num_sample
                    num_turn_sum += num_turn

                    if flag_master:
                        lr_cur = optimizer.state_dict()['param_groups'][0]['lr']
                        pbar.set_description(
                            f'train | batch: [{cnt_batch_train_node} / {tot_batch_train_node}] '
                            f'| loss: {loss_sum/num_sample_sum:.2f} '
                            f'| ce: {loss_ce_sum/num_sample_sum:.2f}  '
                            f'| kl: {loss_kl_sum/num_sample_sum:.2f} '
                            f'| lr: {lr_cur:.2e} '
                        )

                    #----------
                    # 梯度累积（只有最后一步反向传播）
                    loss /= accumulation
                    if level_cuda == 2 and cnt_batch_train_node % accumulation > 0 :
                        context = model.no_sync
                    else:
                        context = nullcontext

                    # 混合精度
                    if use_amp:
                        with context():
                            scaler.scale(loss).backward()
                        if cnt_batch_train_node % accumulation == 0:
                            clip_grad_norm_(model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)  
                            scheduler.step()
                    else:
                        with context():
                            loss.backward()
                        if cnt_batch_train_node % accumulation == 0:
                            clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()  
                            scheduler.step()

                #----------
                # 记录此 epoch 的 Loss
                loss_avg = loss_sum / num_sample_sum
                loss_ce_avg = loss_ce_sum / num_sample_sum
                loss_kl_avg = loss_kl_sum / num_sample_sum
                lst_loss_avg.append([i_epoch, round(loss_avg, 4), round(loss_ce_avg, 4), round(loss_kl_avg, 4)])

            #----------
            # epoch save
            if flag_master:
                if level_cuda == 2:
                    model_save = model.module
                else:
                    model_save = model

                dir_model = sep.join([dir_exp, 'model'])
                #dir_model = sep.join(['..', '..', '..', '..', '..', 'data', 'shenweizhou2', 'ubar_model', 'exp2-ubar-mask'])
                initDir(dir_model)
                dir_pack = sep.join([dir_model, f'epoch{i_epoch}_domain-{args.str_exp_domains}'])
                initDir(dir_pack)

                model_save.save_pretrained(dir_pack)
                tokenizer.save_pretrained(dir_pack)
                pbar.write(f'save: {dir_pack}')

            #----------
            # epoch log
            if flag_master:
                pbar.update(1)
                if i_epoch == max_epoch:
                    pbar.close()

            time_delta = time.time() - time_cur
            if flag_epoch_train and flag_master:
                pbar.write(f'epoch {i_epoch} finish ({time_delta:.2f} s): train loss: {loss_avg:.4f}')

        #--------------------
        # 保存此次训练的各轮损失、各次验证分数
        str_metric_train = str_arg_all
        
        # 所有轮结束，输出训练损失
        logInfo(logger, 'train loss:')
        str_metric_train += 'train loss: \n'
        for i in range(len(lst_loss_avg)):
            i_epoch, loss_avg, loss_ce_avg, loss_kl_avg = lst_loss_avg[i]
            logInfo(logger, f'epoch {i_epoch} | loss: {loss_avg} | ce: {loss_ce_avg} | kl: {loss_kl_avg}')
            str_metric_train += f'epoch {i_epoch} | loss: {loss_avg} | ce: {loss_ce_avg} | kl: {loss_kl_avg} \n'
        logInfo(logger, '----------------------------------------')
        
        return
    
    #----------------------------------------
    # 在 batch 内所有句子的左边填充
    def padList(self, lst_s, pad_token_id, mode='left'):
        
        lst_len_s = [len(s) for s in lst_s]
        maxlen_s = max(lst_len_s)
        lst_len_pad = [maxlen_s - len_s for len_s in lst_len_s]
        lst_pad = [[pad_token_id] * len_pad for len_pad in lst_len_pad]
        lst_front, lst_back = lst_pad, lst_s
        if mode != 'left':
            lst_front, lst_back = lst_back, lst_front
        lst_result = [lst_front[i] + lst_back[i] for i in range(len(lst_s))]
        # maxlen_s: int | batch 内最长句子的长度
        # lst_result: List[List[int]] | 在左边填充后的 batch

        return maxlen_s, lst_result
    
    #----------------------------------------
    # 更新“是否用生成的替换上下文”的标志
    def updateFlagGenElement(self, name_element, p_gen, dependency, flag_gen_element):        
        
        if dependency in ['same']:
            # 相同：b要a要，b不要a不要
            if name_element in ['bspn']:
                flag_gen_element = random.random() < p_gen
        elif dependency in ['random']:
            # 随机：都能决定，独立
            if name_element in ['bspn', 'aspn']:
                flag_gen_element = random.random() < p_gen
        elif dependency in ['only_b']:
            # 只要状态：b决定a不要
            if name_element in ['bspn']:
                flag_gen_element = random.random() < p_gen
            elif name_element in ['aspn']:
                flag_gen_element = False
        elif dependency in ['only_a']:
            # 只要动作：b不要a决定
            if name_element in ['aspn']:
                flag_gen_element = random.random() < p_gen
        elif dependency in ['most_one']:
            # 最多一：b要a不要，b不要a决定
            if name_element in ['bspn']:
                flag_gen_element = random.random() < p_gen
            if name_element in ['aspn']:
                if flag_gen_element:
                    flag_gen_element = False
                else:
                    flag_gen_element = random.random() < p_gen

        return flag_gen_element
