import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

from typing import Any, Optional

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

class PANDataset(Dataset):

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.df.iloc[idx,0], self.df.iloc[idx,1][0], self.df.iloc[idx,2] )

class TokenizerCollate:
    def __init__(self):
        self.tkz = RobertaTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.tkz.add_tokens("xxPERIOD")
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        seqs, targs= batch_split[0], batch_split[1]
        encode = self.tkz(list(seqs), padding="longest")
        return torch.tensor(encode["input_ids"]), torch.tensor(encode["attention_mask"]), torch.tensor(targs)