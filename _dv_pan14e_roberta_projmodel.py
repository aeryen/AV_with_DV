#%%
import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

from typing import Any, Optional

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

from pytorch_lightning.metrics.classification import Accuracy, F1

#%%
class PANDataset(Dataset):

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # return (self.df.iloc[idx,0], self.df.iloc[idx,1][0], self.df.iloc[idx,2])
        return (self.df.iloc[idx,0], self.df.iloc[idx,1], self.df.iloc[idx,2])

class TokenizerCollate:
    def __init__(self, tkz):
        self.tkz = tkz
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        labels, known, unknown = batch_split[0], batch_split[1], batch_split[2]
        labels = np.array(labels) == "Y"
        encode_kno = self.tkz(list(known), truncation=True, padding="max_length", max_length=128)
        encode_unk = self.tkz(list(unknown), truncation=True, padding="max_length", max_length=128)
        return torch.tensor(labels), \
                torch.tensor(encode_kno["input_ids"], dtype=torch.int64), \
                torch.tensor(encode_kno["attention_mask"], dtype=torch.int64), \
                torch.tensor(encode_unk["input_ids"], dtype=torch.int64), \
                torch.tensor(encode_unk["attention_mask"], dtype=torch.int64)
                
# # %%
# dataset_train = PANDataset('./data_pickle_cutcombo/pan_14e_cls/train_essays.pickle')
# tkz = RobertaTokenizer.from_pretrained("roberta-base")
# collator = TokenizerCollate(tkz=tkz)
# dl = DataLoader(dataset_train,
#             batch_size=4,
#             collate_fn=collator,
#             num_workers=0,
#             pin_memory=True, drop_last=False, shuffle=False)
# # %%
# batch = next(iter(dl))
# #%%
# tkz.convert_ids_to_tokens(batch[1][0,:])

# %%
class DVProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.BatchNorm1d(126)
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        self.proj1 = nn.Linear(768, 12)

        self.ln2 = nn.BatchNorm1d(24)
        self.drop2 = nn.Dropout(p=0.1, inplace=True)
        self.dense2 = nn.Linear(24, 24)

        self.ln3 = nn.BatchNorm1d(24)
        self.drop3 = nn.Dropout(p=0.1, inplace=True)
        self.dense3 = nn.Linear(24, 1)

        # self.bias = nn.Parameter( torch.zeros([1]) )

        self.avg = AverageEmbedding()

    def forward(self, kno_dv, kno_mask, unk_dv, unk_mask):
        # dv = [batch, seq_len, 768]

        kno_dv_proj = self.proj1( self.drop1( self.ln1(kno_dv) ) )
        unk_dv_proj = self.proj1( self.drop1( self.ln1(unk_dv) ) )
        # dv_proj = [batch, seq_len, 12]
        
        # along seq_len dim
        kno_dv_proj = self.avg(kno_dv_proj, kno_mask)
        unk_dv_proj = self.avg(unk_dv_proj, unk_mask)
        # kno_dv_proj = [batch, 12]
        
        # method 3, real NN
        dv_comb = torch.cat((kno_dv_proj, unk_dv_proj), dim=1) # [batch, 24]
        dv_comb = F.tanh(dv_comb)

        hidden = self.drop2( self.ln2(dv_comb) )
        hidden = self.dense2(hidden)
        hidden = F.tanh(hidden)  # [batch, 24]

        hidden = self.drop3( self.ln3(hidden) )
        hidden = self.dense3(hidden)  # [batch, 1]

        return hidden

class AverageEmbedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, embedding: torch.Tensor, mask: torch.Tensor):
        embedding = embedding * mask.unsqueeze(-1).float()
        embedding = embedding.sum(1)

        lengths = mask.long().sum(-1)
        length_mask = (lengths > 0)
        # Set any length 0 to 1, to avoid dividing by zero.
        lengths = torch.max(lengths, lengths.new_ones(1))
        # normalize by length
        embedding = embedding / lengths.unsqueeze(-1).float()
        # set those with 0 mask to all zeros, i think
        embedding = embedding * (length_mask > 0).float().unsqueeze(-1)

        return embedding

# %%
class LightningLongformerCLS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_config = config
        
        self.roberta = RobertaForMaskedLM.from_pretrained('roberta-base')
        _ = self.roberta.eval()
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pred_model = self.roberta.roberta
        self.enc_model = self.pred_model.embeddings.word_embeddings
        self.proj_head = DVProjectionHead()

        self.tkz = RobertaTokenizer.from_pretrained("roberta-base")
        self.collator = TokenizerCollate(self.tkz)

        self.lossfunc = nn.BCEWithLogitsLoss()

        self.acc = Accuracy(threshold=0.0)
        self.f1 = F1(threshold=0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])

        return optimizer

    def train_dataloader(self):
        self.dataset_train = PANDataset('./data_pickle_cutcombo/pan_14e_cls/train_essays.pickle')
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_train

    def val_dataloader(self):
        self.dataset_val = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_val

    def test_dataloader(self):
        self.dataset_test = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
        self.loader_test = DataLoader(self.dataset_test,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_test
    
#     @autocast()
    def forward(self, inputs):
        def one_doc_embed(input_ids, input_mask, mask_n=1):
            uniq_mask = []
            uniq_input, inverse_indices = torch.unique( input_ids, return_inverse=True, dim=0 )
            invi = inverse_indices.detach().cpu().numpy()
            for i in range( uniq_input.shape[0] ):
                first_index = np.where(invi == i)[0][0]
                uniq_mask.append(input_mask[first_index,:])

            input_ids = uniq_input
            input_mask = torch.stack(uniq_mask, dim=0)

            embed = self.enc_model(input_ids)

            result_embed = []
            result_pred = []
            # skip start and end symbol
            masked_ids = input_ids.clone()
            for i in range(1, input_ids.shape[1]-mask_n):
                masked_ids[:,i:(i+mask_n)] = self.tkz.mask_token_id

                output = self.pred_model(input_ids=masked_ids, attention_mask=input_mask, return_dict=False)[0]
                result_embed.append( embed[:, i:(i+mask_n), :] )
                result_pred.append( output[:, i:(i+mask_n), :] )

                masked_ids[:,i:(i+mask_n)] = input_ids[:,i:(i+mask_n)]

            # stack along doc_len
            result_embed = torch.cat(result_embed, dim=1)
            result_pred = torch.cat(result_pred, dim=1)

            rec_embed = []
            rec_pred = []
            for i in invi:
                rec_embed.append(result_embed[i,:,:])
                rec_pred.append(result_pred[i,:,:])

            rec_embed = torch.stack(rec_embed, dim=0)
            rec_pred = torch.stack(rec_pred, dim=0)
            return rec_embed, rec_pred

        labels, kno_ids, kno_mask, unk_ids, unk_mask = inputs

        kno_embed, kno_pred = one_doc_embed(input_ids=kno_ids, input_mask=kno_mask)
        unk_embed, unk_pred = one_doc_embed(input_ids=unk_ids, input_mask=unk_mask)

        kno_dv = kno_pred - kno_embed
        unk_dv = unk_pred - unk_embed

        logits = self.proj_head(kno_dv, kno_mask[:,1:-1], unk_dv, unk_mask[:,1:-1])

        logits = torch.squeeze(logits)
        labels = labels.float()
        loss = self.lossfunc(logits, labels)

        return (loss, logits, (kno_embed, kno_pred, unk_embed, unk_pred))
    
    def training_step(self, batch, batch_idx):
        labels, kno_ids, kno_mask, unk_ids, unk_mask  = batch
        
        loss, logits, outputs = self( (labels, kno_ids, kno_mask, unk_ids, unk_mask) )
        
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        labels, kno_ids, kno_mask, unk_ids, unk_mask  = batch
        
        loss, logits, outputs = self( (labels, kno_ids, kno_mask, unk_ids, unk_mask) )
        
        self.acc(logits, labels.float())
        self.f1(logits, labels.float())
        
        return {"val_loss": loss}
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log('eval accuracy', self.acc.compute())

    # def on_test_epoch_start(self):
    #     self.test_logit_outputs = []
    #     self.test_aspect_outputs = []

    # def test_step(self, batch, batch_idx):
    #     input_ids, mask, label  = batch[0].type(torch.int64), batch[1].type(torch.int64), batch[2].type(torch.int64)
        
    #     loss, logits, outputs, aspect_doc = self(input_ids=input_ids, attention_mask=mask, labels=label)
    #     accs = [m(logits, label) for m in self.metrics]  # update metric counters

    #     self.test_logit_outputs.append(logits)
    #     self.test_aspect_outputs.extend(aspect_doc)
        
    #     return loss

    # def on_test_epoch_end(self):
    #     for i,m in enumerate(self.metrics):
    #         print('acc'+str(i), m.compute())


# %%
# _ = model.to("cuda:6")
# %%
# train_dl = model.train_dataloader()
# batch = next(iter(train_dl))
# for i in range(len(batch)):
#     batch[i] = batch[i].to("cuda:6")
# output = model(batch)

# %%
@rank_zero_only
def wandb_save(wandb_logger, train_config):
    wandb_logger.log_hyperparams(train_config)
    wandb_logger.experiment.save('./_dv_pan14e_roberta_projmodel', policy="now")

# %%
if __name__ == "__main__":
    train_config = {}
    train_config["cache_dir"] = "./cache/"
    train_config["epochs"] = 16
    train_config["batch_size"] = 256
    # train_config["accumulate_grad_batches"] = 12
    train_config["gradient_clip_val"] = 1.5
    train_config["learning_rate"] = 5e-5

    pl.seed_everything(42)

    wandb_logger = WandbLogger(name='first_projection',project='AVDV')
    wandb_save(wandb_logger, train_config)

    model = LightningLongformerCLS(train_config)
    # model = LightningLongformerCLS.load_from_checkpoint("./AVDV/143ui570/checkpoints/epoch=5-step=1979.ckpt", config=train_config)
    
    cp_valloss = ModelCheckpoint(save_top_k=5, monitor='val_loss', mode='min')
    trainer = pl.Trainer(max_epochs=train_config["epochs"],
                        # accumulate_grad_batches=train_config["accumulate_grad_batches"],
                        accumulate_grad_batches=1,
                        gradient_clip_val=train_config["gradient_clip_val"],

                        gpus=[6],
                        num_nodes=1,
                        # accelerator='ddp',
                        # plugins='ddp_sharded',

                        amp_backend='native',
                        precision=16,

                        logger=wandb_logger,
                        log_every_n_steps=1,

                        limit_val_batches=40,
                        checkpoint_callback=cp_valloss
                        )

    trainer.fit(model)