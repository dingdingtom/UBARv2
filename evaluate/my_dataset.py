# 使用pytorch dataset封装 all batches
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    '''
    In MyDataset, we view a batch(in all batches) as a sample
    '''
    def __init__(self, all_batches):
        self.all_batches = all_batches

    def __len__(self) -> int:
        # returns the number of batches 
        return len(self.all_batches)
    
    def __getitem__(self, index: int):
        # returns a batch from self.all_batches
        return self.all_batches[index]


'''
old:
self.model


'''

        