# 导入库
from init import *

#----------------------------------------
# 训练输入参数解释器
def getParserTrain():
    
    parser = argparse.ArgumentParser(prog='train', description='UBAR training')
    
    #--------------------
    # 参数组：设备
    group_device = parser.add_argument_group(title='group device', description='options for device')
    group_device.add_argument('--id_cuda_single', default=4, type=int, help='id cuda when `level_cuda` is 1 [default: 0]')
    group_device.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed train [default: -1]')
    group_device.add_argument('--use_cpu',default=False, type=bool, help='whether use cpu [default: False]')

    #--------------------
    # 参数组：实验
    group_experiment = parser.add_argument_group(title='group experiment', description='options for experiment')
    # sep.join(['experiments_all', 'all_sd-0_lr-0.0001_bs-2_am-4', 'model', 'epoch25_inform-94.48%_success-82.85%_bleu-15.43_score-104.09'])
    group_experiment.add_argument('--dir_pack', default='', type=str, help='dir to model pack [default: ]')
    group_experiment.add_argument('--log_console', default=True, type=bool, help='whether show train epoch [default: False]')
    group_experiment.add_argument('--mode', default='train', type=str, choices=['train', 'train2'], help='mode for run [default: train]')
    group_experiment.add_argument('--num_low_resource', default=10, type=int, help='rate use low resource')
    group_experiment.add_argument('--save_model', default=True, type=bool, help='whether save model after validation [default: True]')
    group_experiment.add_argument('--str_exp_domains', default='all', type=str, help='exp domains use `_` to spilt (except_hotel, hotel) [default: all]')
    group_experiment.add_argument('--use_low_resource', default=0, type=int, help='whether use low resource')

    #--------------------
    # 参数组：模型优化
    group_learn = parser.add_argument_group(title='group learn', description='options for model learning')
    group_learn.add_argument('--accumulation', default=8, type=int, help='accumulate gradient to stimulate training with big batch size [default: 16]')
    group_learn.add_argument('--batch_size_train', default=1, type=int, help='batch size for train [default: 2]')
    group_learn.add_argument('--lr', default=0.00015, type=float, help='initial learning rate [default: 1e-4]')
    group_learn.add_argument('--max_epoch', default=75, type=int, help='max number of epochs for train [default: 20]')
    group_learn.add_argument('--max_grad_norm', default=5, type=int, help='grad norm cutoff to prevent explosion of gradient [default: 200]')
    group_learn.add_argument('--num_step_warmup', default=-1, type=int, help='number of step for warmup [default: -1]')
    group_learn.add_argument('--seed', default=0, type=int, help='seed for train [default: 0]')
    group_learn.add_argument('--type_optimizer', default='adamw', type=str, choices=['adam', 'adamw'], help='optimizer for train [default: adamw]')
    group_learn.add_argument('--type_scheduler', default='cosw', type=str, choices=['constw', 'cosw', 'linw', 'polyw'], help='scheduler for train [default: constw]')
    group_learn.add_argument('--use_amp', default=False, type=bool, help='whether use amp [default: True]')
    group_learn.add_argument('--weight_decay', default=0, type=float, help='weight decay for AdamW optimizer [default: 0]')
    
    #--------------------
    # 参数组：模型超参数
    group_model = parser.add_argument_group(title='group model', description='options for model details')
    group_model.add_argument('--p_mask_input1', default=0, type=float, help='probability mask input1 [default: 0.2]')
    group_model.add_argument('--p_mask_input2', default=0, type=float, help='probability mask input1 [default: 0.2]')
    group_model.add_argument('--p_mask_label', default=0, type=float, help='probability drop label [default: 0.2]')
    group_model.add_argument('--rate_kl', default=1e-2, type=float, help='rate for kl loss against ce loss [default: 1e-3]')
    group_model.add_argument('--str_name_element_mask_input1', default='bspn', type=str, help='name element needed to mask for masked-LM [default: none]')
    group_model.add_argument('--str_name_element_mask_input2', default='bspn', type=str, help='name element needed to mask for masked-LM [default: none]')
    group_model.add_argument('--str_name_element_mask_label', default='user_db', type=str, help='name element needed to drop before ce loss [default: none]')
    group_model.add_argument('--target_mask', default='none', type=str, choices=['none', 'input', 'label', 'input_label'], help='mask target during forward  [default: input]')
    group_model.add_argument('--use_same_mask', default=0, type=int, help='whether only mask two input in a same way for R-Drop [default: False]')
    
    return parser
