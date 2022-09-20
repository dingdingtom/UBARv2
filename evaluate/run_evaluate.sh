#!/bin/bash
# e2e modeling
# generated r, lr=5e-5
# 设置模型保存路径
export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=4

load_path='../train/experiments_all2/all_sd-0_lr-0.00015_bs-1_am-8/model/epoch8_domain-all'
python -m torch.distributed.launch --nproc_per_node=1 evaluation.py -mode test -cfg eval_load_path=$load_path batch_size=4 multi_gpu=True use_true_prev_bspn=True use_true_prev_aspn=False use_true_db_pointer=True use_true_prev_resp=False use_true_curr_bspn=True use_true_curr_aspn=False use_all_previous_context=True && echo $load_path is done
