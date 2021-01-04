# %%
import os
from fastai.text.data import TextDataLoaders
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from fastai.text.all import *

# %%

def get_lm_dataloader():
    print( "Init LM DataLoader" )
    df_train = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/train_essays.pickle')
    df_test01 = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/test01_essays.pickle')
    df_test02 = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/test02_essays.pickle')
    uniq_list = []
    for dfi, df in enumerate([df_train, df_test01, df_test02]):
        for i in range(len(df)):
            for doc in df.iloc[i, 1]:
                if(doc not in uniq_list):
                    uniq_list.append(doc)
            if(df.iloc[i, 2] not in uniq_list):
                uniq_list.append(df.iloc[i, 2])
    print( "Unique doc list of len:", len(uniq_list) )
    uniq_list = pd.DataFrame(uniq_list)
    dl = TextDataLoaders.from_df(uniq_list,
                                path='./data_pickle_fastai/pan_14e_cls/',
                                is_lm=True,
                                text_col=0,
                                valid_pct=0.15
                                )
    print( "LM DataLoader Done." )
    return dl

def get_cls_dataloader():
    print( "Init CLS DataLoader" )
    df_train = pd.read_pickle('./data_new/pan_14e_cls/train_essays.pickle')
    df_test01 = pd.read_pickle('./data_new/pan_14e_cls/test01_essays.pickle')
    # df_test02 = pd.read_pickle('./data_new/pan_14e_cls/test02_essays.pickle')
    train_list = []
    for i in range(len(df_train)):
        for doc in df_train.iloc[i, 1]:
            train_list.append((df_train.iloc[i, 0], doc, df_train.iloc[i, 2]))

    valid_list = []
    for i in range(len(df_test01)):
        for doc in df_test01.iloc[i, 1]:
            valid_list.append((df_test01.iloc[i, 0], doc, df_test01.iloc[i, 2]))

    print( "TRAIN VALID size:", len(train_list), len(valid_list) )

    train_df = pd.DataFrame(train_list)
    train_df["is_valid"] = False
    valid_df = pd.DataFrame(valid_list)
    valid_df["is_valid"] = True

    all_df = pd.concat([train_df, valid_df])
    dl = TextDataLoaders.from_df(all_df,
                                path='./data_new/pan_14e_cls/',
                                label_col=0,
                                text_col=1,
                                valid_col="is_valid"
                                )
    # print( "LM DataLoader Done." )
    return dl

# %%
lmdl = get_lm_dataloader()
learn = language_model_learner(lmdl, AWD_LSTM)

# %%
learn.predict('This movie is about', n_words=20)
# %%
learn.model
# %%
def one_doc_embed(learn, doc):
    pred_enc = model[0]
    doc_enc = model[0].encoder
    
    ids, _ = learn.data.one_item(doc)
    
    with torch.no_grad():
        pred_enc.reset()
        embed = doc_enc(ids).detach()
        pred  = pred_enc(ids)[0][-1].detach()
        
        return {"e":embed, "p":pred}

def pred_raw(model, df):
    prob_list = []
    label_list = []
    
    for i in tqdm(range(len(df))):
        k_results = []
        for k_doc in df.iloc[i,1]:
            k_results.append( one_doc_embed(model, k_doc) )

        u_results = one_doc_embed(learn, df.iloc[i,2])
        prob_list.append( {"k":k_results, "u":u_results} )
        label_list.append( df.iloc[i,0] )
        
    return  prob_list, label_list

# %%
df_train = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/train_essays.pickle')
df_test01 = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/test01_essays.pickle')
df_test02 = pd.read_pickle('./data_pickle_fastai/pan_14e_cls/test02_essays.pickle')

# %%
test_prob, test_label = pred_raw(learn, df_test02)

# %%

# %%
