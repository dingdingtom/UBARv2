# 导入库
from init import *

from init_logger import *
from model_ubar2 import ModelUbar2
from init_parser2 import getParserTrain
from config import config_global as cfg

#----------------------------------------
# 训练
def train(args):
    
    accumulation = args.accumulation
    batch_size_train = args.batch_size_train
    flag_master = args.flag_master
    level_cuda = args.level_cuda
    local_rank = args.local_rank
    log_console = args.log_console
    lr = args.lr
    seed = args.seed
    
    #--------------------
    # 指定领域 
    dir1_exp = './experiments_all2' if 'all' in cfg.exp_domains else './experiments_part2'
    dir2_exp = f'{"-".join(cfg.exp_domains)}_sd-{seed}_lr-{lr}_bs-{batch_size_train}_am-{accumulation}'
    dir_exp = sep.join([dir1_exp, dir2_exp])
    if flag_master:
        initDir(dir1_exp)
        initDir(dir_exp)
    cfg.dir_exp = dir_exp
        
    #--------------------
    # 初始化日志
    logger, ch, fh = getLogger(level_cuda, dir_exp, 'train', log_console)

    #----------
    # 运行环境
    logInfo(logger, f'dir experiments: {dir_exp}')
    logInfo(logger, f'torch: {version_torch}')
    logInfo(logger, f'cuda: {version_cuda}')
    logInfo(logger, f'cudnn: {version_cudnn}')
    logInfo(logger, f'cuda available: {cuda_available}')
    logInfo(logger, f'num device: {num_device}')
            
    #--------------------
    # 模型
    obj_model = ModelUbar2(args, logger)
    model = obj_model.model
            
    #--------------------
    # DDP 模型
    if level_cuda == 2:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    #--------------------
    # 训练
    obj_model.model = model
    obj_model.train()
    
    return

#----------------------------------------
# 主函数
def main():
    
     # 解析输入参数
    parser = getParserTrain()
    args = parser.parse_args()

    id_cuda_single = args.id_cuda_single
    local_rank = args.local_rank
    seed = args.seed
    str_name_element_mask_input1 = args.str_name_element_mask_input1
    str_name_element_mask_input2 = args.str_name_element_mask_input2
    str_name_element_mask_label = args.str_name_element_mask_label
    str_coef_p_gen_train = args.str_coef_p_gen_train
    use_cpu = args.use_cpu
    
    cfg.num_device = num_device

    #--------------------
    # GPU
    if use_cpu:
        level_cuda = 0
    elif cuda_available:
        if local_rank == -1:
            level_cuda = 1
        else:
            level_cuda = 2
    args.level_cuda = level_cuda
    
    #--------------------
    # 设备
    if level_cuda == 0:
        device = torch.device('cpu')
    elif level_cuda == 1:
        device = torch.device(f'cuda:{id_cuda_single}')
    elif level_cuda == 2:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    # 主节点
    flag_master = level_cuda in [0, 1] or (level_cuda == 2 and dist.get_rank() == 0)
    args.device = device
    args.flag_master = flag_master
        
    #--------------------
    # 种子
    rank = 0 if level_cuda in [0, 1] else dist.get_rank()
    setSeed(seed + rank, level_cuda)
    
    #--------------------
    # 训练
    cfg.mode = args.mode
    cfg.exp_domains = args.str_exp_domains.split('_')
    args.lst_name_element_mask_input1 = str_name_element_mask_input1.split('_')
    args.lst_name_element_mask_input2 = str_name_element_mask_input2.split('_')
    args.lst_name_element_mask_label = str_name_element_mask_label.split('_')
    
    lst_name_element_gen = ['bspn', 'aspn', 'resp']
    args.dic_element_name2coef_train = \
        {lst_name_element_gen[i]: float(str_coef_p_gen_train.split('_')[i]) for i in range(3)}

    train(args)

    return

#----------------------------------------
# 入口
if __name__ == '__main__':
    main()
    