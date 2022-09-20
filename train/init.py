# 导入库
import os
import re
import csv
import sys
import json
import math
import time
import spacy
import random
import string
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from nltk.util import ngrams
from itertools import product
from collections import Counter

import ontology

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn
from torch import optim
from torch import LongTensor
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast as autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import optimization
from transformers.optimization import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

version_str = sys.version
version = [int(i) for i in version_str.split('.')[:2]]
if version[0] >= 3 and version[1] >= 7:
    from contextlib import nullcontext
else:
    from contextlib import suppress as nullcontext

#----------------------------------------
# 常量
sep = os.sep
e = math.exp(1)
epsilon = 1e-8
nlp = spacy.load('en_core_web_sm')

version_torch = torch.__version__
version_cuda = torch.version.cuda
version_cudnn = torch.backends.cudnn.version()
cuda_available = torch.cuda.is_available()
num_device = torch.cuda.device_count()

warnings.filterwarnings('ignore')

#----------------------------------------
# 设置种子
def setSeed(seed=1, level_cuda=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if level_cuda != 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

#----------------------------------------
# 创建文件夹
def initDir(dir_tgt):
    if not os.path.exists(dir_tgt):
        os.mkdir(dir_tgt)
    return
