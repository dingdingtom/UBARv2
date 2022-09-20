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
    # sep.join(['experiments_all', 'all_sd-0_lr-0.00015_bs-2_am-4', 'model', 'epoch75_inform-94.09%_success-83.08%_bleu-15.73_score-104.32'])
    group_experiment.add_argument('--dir_pack', default=sep.join(['experiments_part', 'taxi_sd-0_lr-0.00015_bs-1_am-8', 'model', 'epoch2-domain-taxi']), type=str, help='dir to model pack [default: ]')
    group_experiment.add_argument('--log_console', default=True, type=bool, help='whether show train epoch [default: False]')
    group_experiment.add_argument('--mode', default='train2', type=str, choices=['train', 'train2'], help='mode for run [default: train]')
    group_experiment.add_argument('--num_low_resource', default=10, type=int, help='rate use low resource')
    group_experiment.add_argument('--save_model', default=True, type=bool, help='whether save model after validation [default: True]')
    group_experiment.add_argument('--str_exp_domains', default='all', type=str, help='exp domains use `_` to spilt (except_hotel, hotel) [default: all]')
    group_experiment.add_argument('--use_low_resource', default=0, type=int, help='whether use low resource')

    #--------------------
    # 参数组：模型优化
    group_learn = parser.add_argument_group(title='group learn', description='options for model learning')
    group_learn.add_argument('--accumulation', default=8, type=int, help='accumulate gradient to stimulate training with big batch size [default: 16]')
    group_learn.add_argument('--batch_size_train', default=1, type=int, help='batch size for train [default: 2]')
    group_learn.add_argument('--lr', default=0.000015, type=float, help='initial learning rate [default: 1e-4]')
    group_learn.add_argument('--max_epoch', default=10, type=int, help='max number of epochs for train [default: 20]')
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
    group_model.add_argument('--dependency', default='only_b', type=str, choices=['same', 'random', 'only_b', 'only_a', 'most_one'], help='generation relationship between belief and action [default: same]')
    group_model.add_argument('--num_beams_train', default=1, type=int, help='number of beam for train generation [default: 1]')
    group_model.add_argument('--p_gen_lower', default=0.01, type=float, help='probability lower bound for using generation to replace context [default: 0.06]')
    group_model.add_argument('--p_gen_upper', default=0.01, type=float, help='probability upper bound for using generation to replace context [default: 0.07]')
    group_model.add_argument('--p_mask_input1', default=0.02, type=float, help='probability mask input1 [default: 0.2]')
    group_model.add_argument('--p_mask_input2', default=0.02, type=float, help='probability mask input1 [default: 0.2]')
    group_model.add_argument('--p_mask_label', default=0, type=float, help='probability drop label [default: 0.2]')
    group_model.add_argument('--rate_kl', default=1e-2, type=float, help='rate for kl loss against ce loss [default: 1e-3]')
    group_model.add_argument('--str_coef_p_gen_train', default='1_1_1', type=str, help='coef probability for belief action response during train generation  [default: 1_0_0]')
    group_model.add_argument('--str_name_element_mask_input1', default='bspn', type=str, help='name element needed to mask for masked-LM [default: none]')
    group_model.add_argument('--str_name_element_mask_input2', default='bspn', type=str, help='name element needed to mask for masked-LM [default: none]')
    group_model.add_argument('--str_name_element_mask_label', default='user_db', type=str, help='name element needed to drop before ce loss [default: none]')
    group_model.add_argument('--top_p_train', default=1, type=float, help='top probability threshold for train generation [default: 1]') 
    group_model.add_argument('--target_mask', default='none', type=str, choices=['none', 'input', 'label', 'input_label'], help='mask target during forward  [default: input]')
    group_model.add_argument('--use_gen_mask', default=0, type=int, help='whether only mask generation part of second input for R-Drop [default: False]')
    group_model.add_argument('--use_raw_cat', default=0, type=int, help='whether use raw concatenate in element generation [default: False]')
    group_model.add_argument('--use_raw_context', default=0, type=int, help='whether use raw context in element generation [default: False]')
    group_model.add_argument('--use_same_mask', default=0, type=int, help='whether only mask two input in a same way for R-Drop [default: False]')

    return parser
