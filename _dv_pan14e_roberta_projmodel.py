#%%
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

#%%
class PANDataset(Dataset):

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.df.iloc[idx,0], self.df.iloc[idx,1][0], self.df.iloc[idx,2] )

class TokenizerCollate:
    def __init__(self):
        self.tkz = RobertaTokenizer.from_pretrained("roberta-base")
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        labels, unknown, known = batch_split[0], batch_split[1], batch_split[2]
        labels = np.array(labels) == "Y"
        encode_unk = self.tkz(list(unknown), truncation=True, padding="max_length", max_length=256)
        encode_kno = self.tkz(list(known), truncation=True, padding="max_length", max_length=256)
        return torch.tensor(labels), torch.tensor(encode_unk["input_ids"]), torch.tensor(encode_unk["attention_mask"]), \
                torch.tensor(encode_kno["input_ids"]), torch.tensor(encode_kno["attention_mask"])


dataset_train = PANDataset('./data_pickle_trfm/pan_14e_cls/train_essays.pickle')
collator = TokenizerCollate()
dl = DataLoader(dataset_train,
            batch_size=4,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True, drop_last=False, shuffle=False)

batch = next(iter(dl))
# %%
collator.tkz.convert_ids_to_tokens(batch[1][0,:])

# %%
class LightningLongformerCLS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_config = config
        
        self.roberta = RobertaForMaskedLM.from_pretrained('roberta-base')
        _ = self.roberta.eval()
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.lossfunc = MultiLabelCEL()
        self.metrics = torch.nn.ModuleList( [AspectACC(aspect=i) for i in range(6)] )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])

        return optimizer

    def train_dataloader(self):
        dataset_train = PANDataset('./data_pickle_trfm/pan_14e_cls/train_essays.pickle')
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.tokenCollate,
                                        num_workers=2,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_train

    def val_dataloader(self):
        dataset_train = PANDataset('./data_pickle_trfm/pan_14e_cls/test01_essays.pickle')
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.tokenCollate,
                                        num_workers=2,
                                        pin_memory=True, drop_last=False, shuffle=True)
        return self.loader_val

    def test_dataloader(self):
        self.dataset_test = ReviewDataset("../../data/hotel_balance_LengthFix1_3000per/df_test.pickle")
        self.loader_test = DataLoader(self.dataset_test,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.tokenCollate,
                                        num_workers=0,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_test
    
#     @autocast()
    def forward(self, input_ids, attention_mask, labels):
        logits,outputs,aspect_doc = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.lossfunc(logits, labels)

        return (loss, logits, outputs, aspect_doc)
    
    def training_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        self.log("train_loss", loss)
        
        return loss

    def on_after_backward(self):
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'] )
        with torch.no_grad():
            if (model.longformer.longformer.embeddings.word_embeddings.weight.grad is not None):
                norm_value = model.longformer.longformer.embeddings.word_embeddings.weight.grad.detach().norm(2).item()
                self.log('NORMS/embedding norm', norm_value)

            for i in [0, 4, 8, 11]:
                if (model.longformer.longformer.encoder.layer[i].output.dense.weight.grad is not None):
                    norm_value = model.longformer.longformer.encoder.layer[i].output.dense.weight.grad.detach().norm(2).item()
                    self.log('NORMS/encoder %d output norm' % i, norm_value)

            if (self.longformer.classifier.aspect_projector[2].weight.grad is not None):
                norm_value = self.longformer.classifier.aspect_projector[2].weight.grad.detach().norm(2).item()
                self.log("NORMS/aspect_projector", norm_value)

            if (self.longformer.classifier.senti_projector[2].weight.grad is not None):
                norm_value = self.longformer.classifier.senti_projector[2].weight.grad.detach().norm(2).item()
                self.log("NORMS/senti_projector", norm_value)

            if (self.longformer.classifier.aspect[2].weight.grad is not None):
                norm_value = self.longformer.classifier.aspect[2].weight.grad.detach().norm(2).item()
                self.log("NORMS/aspect", norm_value)

            if (self.longformer.classifier.sentiment[2].weight.grad is not None):
                norm_value = self.longformer.classifier.sentiment[2].weight.grad.detach().norm(2).item()
                self.log("NORMS/sentiments", norm_value)

    def validation_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        
        # self.log('val_loss', loss, on_step=False, on_epoch=True, reduce_fx=torch.mean, prog_bar=False)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters
        
        return {"val_loss": loss}
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)

        for i,m in enumerate(self.metrics):
            self.log('acc'+str(i), m.compute())

    def on_test_epoch_start(self):
        self.test_logit_outputs = []
        self.test_aspect_outputs = []

    def test_step(self, batch, batch_idx):
        input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
        loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
        accs = [m(logits, label) for m in self.metrics]  # update metric counters

        self.test_logit_outputs.append(logits)
        self.test_aspect_outputs.extend(aspect_doc)
        
        return loss

    def on_test_epoch_end(self):
        for i,m in enumerate(self.metrics):
            print('acc'+str(i), m.compute())