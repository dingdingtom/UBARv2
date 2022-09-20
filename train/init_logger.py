# 导入库
from init import *

#----------------------------------------
# 初始化日志
def getLogger(level_cuda, dir_exp, name, flag_console=False):
    # 只有一个 logger
    if (level_cuda in [0, 1] or (level_cuda == 2 and dist.get_rank() == 0)):
        # 日志记录所有信息
        logger = logging.getLogger(f'{name} logger')
        logger.setLevel(logging.DEBUG)
        
        # 需要关闭日志传递（因为多卡时会有子日志）
        logger.propagate = False
        
        #--------------------
        # 流 handler 向控制台 console 记录所有信息
        ch = logging.StreamHandler()
        if flag_console:
            ch.setLevel(logging.INFO)
        else: 
            ch.setLevel(logging.ERROR)
        
        #--------------------
        # 文件 handler 向文件记录所有的信息
        dir_log = sep.join([dir_exp, 'log'])
        initDir(dir_log)
            
        fh = logging.FileHandler(
            filename=sep.join([dir_log, f'log_{name}.log']),
            mode='w',
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        
        #--------------------
        # handler 格式
        formatter = logging.Formatter(
            fmt=f'%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        #--------------------
        # 父日志绑定 handler
        logger.addHandler(ch)
        logger.addHandler(fh)
    else:
        logger, ch, fh = None, None, None
        
    return logger, ch, fh

#----------------------------------------
# 输出信息
def logInfo(logger, info):
    if logger is not None:
        logger.info(info)
    return

#----------------------------------------
# 解绑日志
def releaseLogger(logger, ch, fh):
    if logger is not None:
        #--------------------
        # 父日志解绑 handler
        ch.flush()
        fh.close()
        logger.removeHandler(ch)
        logger.removeHandler(fh)
    return
