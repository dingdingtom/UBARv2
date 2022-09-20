# 导入库
from init import *

from config import config_global as cfg

#----------------------------------------
# Session Level 数据集
class DialogDatasetSessionLevel(Dataset):
    
    # 初始化
    def __init__(self, lst_dial):
        
        self.lst_dial = lst_dial
        
        self.loadData()

        return
    
    #--------------------
    # 加载数据
    def loadData(self):
        
        lst_dial = self.lst_dial
        
        #----------
        # 按顺序拼接各个 turn，每个 turn 中的内容按 UBDAR 排序
        lst_name_element = ['user', 'bspn', 'db', 'aspn', 'resp']
        lst_info_context = []
        for dial in lst_dial:
            context = []
            cnt_turn = 0
            for dic_turn in dial:
                cnt_turn += 1
                for name_element in lst_name_element:
                    element = dic_turn[name_element]
                    context += element
            lst_info_context.append([cnt_turn, len(context), context])
        # lst_info_context: List[List[x]]
        #     每个元素是有3个元素的一维 List：该 context (session) 的 turn 个数，词语个数，它的各个词语的下标 (List)
        
        #----------
        self.lst_info_context = lst_info_context
        
        return
    
    #--------------------
    # 加载数据
    def getListBatch(self, batch_size):
        
        lst_info_context = self.lst_info_context
        
        maxlen = cfg.maxlen
        pad_id = cfg.pad_id
        
        #----------
        # 按 turn 的个数归类各个 context
        dic_turn2contexts = {}
        for cnt_turn, len_context, context in lst_info_context:
            dic_turn2contexts.setdefault(cnt_turn, []).append(context)

        # 每个类的 context 个数是 batch_size 的倍数
        lst_batch = []
        tot_session = 0
        tot_turn = 0
        for cnt_turn, lst_context in dic_turn2contexts.items():
            # 丢弃不能整除 batch_size 的部分
            cnt_context = len(lst_context)
            num_drop = cnt_context % batch_size
            if num_drop:
                lst_context = lst_context[:-num_drop] 
            lst_context = [LongTensor(context) for context in lst_context]
            cnt_context -= num_drop
            tot_session += cnt_context
            tot_turn += cnt_turn * cnt_context
            
            # 每 batch_size 个 context 组合成一个 batch，按长度补齐
            for i_context in range(0, cnt_context, batch_size):
                batch = lst_context[i_context: i_context+batch_size]
                batch_pad = pad_sequence(batch, batch_first=True, padding_value=pad_id)
                # batch_pad: Tensor | shape (batch_size, maxlen_batch)
                maxlen_batch = min(1024, batch_pad.shape[1])
                batch_pad = batch_pad[:, :maxlen_batch]
                lst_batch.append(batch_pad)
                
        #----------
        self.lst_batch = lst_batch
        self.tot_session = tot_session
        self.tot_turn = tot_turn
            
        return
    
    #--------------------
    # 数据集长度
    def __len__(self):
        return len(self.lst_batch)
    
    #--------------------
    # 数据集某项
    def __getitem__(self, idx):
        return self.lst_batch[idx]
    
#----------------------------------------
# Turn Level 数据集
class DialogDatasetTurnLevel(Dataset):
    
    # 初始化
    def __init__(self, lst_dial):
        
        self.lst_dial = lst_dial
        
        self.loadData()
        
        return
    
    #--------------------
    # 加载数据
    def loadData(self):
        
        lst_dial = self.lst_dial
        
        #----------
        # 按顺序拼接各个 turn，每个 turn 中的内容按 id domain pointer UBDAR 排序
        lst_name_element = ['dial_id', 'turn_domain', 'pointer', 'user', 'bspn', 'db', 'aspn', 'resp']
        lst_info_context = []
        for dial in lst_dial:
            context = []
            cnt_turn = 0
            for dic_turn in dial:
                dic_turn_part = {}
                cnt_turn += 1
                for name_element in lst_name_element:
                    dic_turn_part[name_element] = dic_turn[name_element][:]
                context.append(dic_turn_part)
            lst_info_context.append([cnt_turn, context])
        # lst_info_context: List[List[x]]
        #     每个元素是有2个元素的一维 List：该 context (session) 的 turn 个数，它的各个 Dict 类型的 turn (List)
        
        #----------
        self.lst_info_context = lst_info_context
        
        return
    
    #--------------------
    # 加载数据
    def getListBatch(self, batch_size):
        
        lst_info_context = self.lst_info_context
        
        maxlen = cfg.maxlen
        pad_id = cfg.pad_id
        
        #----------
        # 按 turn 的个数归类各个 context
        dic_turn2contexts = {}
        for cnt_turn, context in lst_info_context:
            dic_turn2contexts.setdefault(cnt_turn, []).append(context)

        # 每个类的 context 个数是 batch_size 的倍数
        lst_batch = []
        tot_session = 0
        tot_turn = 0
        for cnt_turn, lst_context in dic_turn2contexts.items():
            # 丢弃不能整除 batch_size 的部分
            cnt_context = len(lst_context)
            num_drop = 0#cnt_context % batch_size
            if num_drop:
                lst_context = lst_context[:-num_drop] 
            cnt_context -= num_drop
            tot_session += cnt_context
            tot_turn += cnt_turn * cnt_context

            # 每 batch_size 个 context 组合成一个 batch
            for i_context in range(0, cnt_context, batch_size):
                batch = lst_context[i_context: i_context+batch_size]
                lst_batch.append(batch)
                
        #----------
        self.lst_batch = lst_batch
        self.tot_session = tot_session
        self.tot_turn = tot_turn
            
        return
    
    #--------------------
    # 数据集长度
    def __len__(self):
        return len(self.lst_batch)
    
    #--------------------
    # 数据集某项
    def __getitem__(self, idx):
        return self.lst_batch[idx]
    
#--------------------
# 调整加载的 batch
def collate_fn(batch_dial_raw):

    # batch_dial_raw: List | data loader 一次加载的 batch，设定 `batch_size` 为 1
    batch_dial = batch_dial_raw[0]
    # batch_dial: List[List[Dict]] | 
    #     每个元素是一个对话，对话是列表，它的每个元素是一轮，轮是字典

    return batch_dial
