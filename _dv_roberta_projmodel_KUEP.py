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
from torch.cuda.amp import autocast

import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

from pytorch_lightning.metrics.classification import Accuracy, F1

#%%
class PANDatasetKUEP(Dataset):

    def __init__(self, df_path):
        self.df = torch.load(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.df[idx]["l"], self.df[idx]["k"], self.df[idx]["u"])

class TokenizerCollateKUEP:
    
    def __call__(self, batch):
        batch_split = list(zip(*batch))
        labels, known, unknown = batch_split[0], batch_split[1], batch_split[2]
        labels = np.array(labels)

        kno_ids = torch.stack( [k["input_ids"] for k in known] )
        kno_mask = torch.stack( [k["input_mask"] for k in known] )
        kno_e = torch.stack( [k["e"] for k in known] )
        kno_p = torch.stack( [k["p"] for k in known] )

        unk_ids = torch.stack( [k["input_ids"] for k in unknown] )
        unk_mask = torch.stack( [k["input_mask"] for k in unknown] )
        unk_e = torch.stack( [k["e"] for k in unknown] )
        unk_p = torch.stack( [k["p"] for k in unknown] )
        return torch.tensor(labels), \
                kno_ids, kno_mask, kno_e, kno_p, \
                unk_ids, unk_mask, unk_e, unk_p

# %%
dataset_train = PANDatasetKUEP("./data_pickle_cutcombo/pan_13_cls/train_KUEP_combo.pt")
loader_train = DataLoader(dataset_train,
                                batch_size=4,
                                collate_fn=TokenizerCollateKUEP(),
                                num_workers=1,
                                pin_memory=True, drop_last=False, shuffle=False)


# %%
class DVProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.hid_sz = 12
        self.ln1 = nn.BatchNorm1d(126)
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        self.proj1 = nn.Linear(768, self.hid_sz)

        self.ln2 = nn.BatchNorm1d(self.hid_sz*2)
        self.drop2 = nn.Dropout(p=0.1, inplace=True)
        self.dense2 = nn.Linear(self.hid_sz*2, self.hid_sz*2)

        self.ln3 = nn.BatchNorm1d(self.hid_sz*2)
        self.drop3 = nn.Dropout(p=0.1, inplace=True)
        self.dense3 = nn.Linear(self.hid_sz*2, 1)

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
        dv_comb = torch.tanh(dv_comb)

        hidden = self.drop2( self.ln2(dv_comb) )
        hidden = self.dense2(hidden)
        hidden = torch.tanh(hidden)  # [batch, 24]

        hidden = self.drop3( self.ln3(hidden) )
        hidden = self.dense3(hidden)  # [batch, 1]

        return hidden

class DVProjectionHead_ActiFirst(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj_sz = 12
        self.act_sz = 12
        self.doc_sz = 12

        self.ln1 = nn.BatchNorm1d(126)
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        self.proj1 = nn.Linear(768, self.proj_sz)

        self.lnp = nn.BatchNorm1d(126)
        self.dropp = nn.Dropout(p=0.1, inplace=True)
        self.densep = nn.Linear(self.proj_sz, self.act_sz)

        self.ln2 = nn.BatchNorm1d(self.act_sz*2)
        self.drop2 = nn.Dropout(p=0.1, inplace=True)
        self.dense2 = nn.Linear(self.act_sz*2, self.doc_sz)

        self.ln3 = nn.BatchNorm1d(self.doc_sz)
        self.drop3 = nn.Dropout(p=0.1, inplace=True)
        self.dense3 = nn.Linear(self.doc_sz, 1)

        self.avg = AverageEmbedding()

    def forward(self, kno_dv, kno_mask, unk_dv, unk_mask):
        # dv = [batch, seq_len, 768]

        kno_dv_proj = self.proj1( self.drop1( self.ln1(kno_dv) ) )
        kno_dv_proj = torch.tanh(kno_dv_proj)
        unk_dv_proj = self.proj1( self.drop1( self.ln1(unk_dv) ) )
        unk_dv_proj = torch.tanh(unk_dv_proj)
        # dv_proj = [batch, seq_len, 12]

        kno_dv_proj = self.densep( self.dropp( self.lnp(kno_dv_proj) ) )
        kno_dv_proj = torch.tanh(kno_dv_proj)  # [batch, seq_len, 12]
        unk_dv_proj = self.densep( self.dropp( self.lnp(unk_dv_proj) ) )
        unk_dv_proj = torch.tanh(unk_dv_proj)  # [batch, seq_len, 12]

        # along seq_len dim
        kno_dv_proj = self.avg(kno_dv_proj, kno_mask)  # [batch, 12]
        unk_dv_proj = self.avg(unk_dv_proj, unk_mask)  # [batch, 12]

        dv_comb = torch.cat((kno_dv_proj, unk_dv_proj), dim=1) # [batch, 24]

        hidden = self.dense2( self.drop2( self.ln2(dv_comb) ) )
        hidden = torch.tanh(hidden)  # [batch, 24]

        hidden = self.dense3( self.drop3( self.ln3(hidden) ) )  # [batch, 1]

        return hidden

class DVProjectionHead_EmbActi(nn.Module):

    def __init__(self):
        super().__init__()
        self.hid_sz = 32
        self.tok_feat_sz = 64
        self.doc_feat_sz = 32

        self.dvln1 = nn.BatchNorm1d(126)
        self.dvdrop1 = nn.Dropout(p=0.5, inplace=True)
        self.dvproj1 = nn.Linear(768, self.hid_sz)

        self.embln1 = nn.BatchNorm1d(126)
        self.embdrop1 = nn.Dropout(p=0.5, inplace=True)
        self.embproj1 = nn.Linear(768, self.hid_sz)

        self.tfln = nn.BatchNorm1d(126)
        self.tfdrop = nn.Dropout(p=0.1, inplace=True)
        self.tfdense = nn.Linear(self.hid_sz*2, self.tok_feat_sz)

        self.ln2 = nn.BatchNorm1d(self.tok_feat_sz*2)
        self.drop2 = nn.Dropout(p=0.1, inplace=True)
        self.dense2 = nn.Linear(self.tok_feat_sz*2, self.doc_feat_sz)

        self.ln3 = nn.BatchNorm1d(self.doc_feat_sz)
        self.drop3 = nn.Dropout(p=0.1, inplace=True)
        self.dense3 = nn.Linear(self.doc_feat_sz, 1)

        self.avg = AverageEmbedding()

    def forward(self, kno_emb, kno_dv, kno_mask, unk_emb, unk_dv, unk_mask):
        # dv = [batch, seq_len, 768]

        kno_dv_proj = self.dvproj1( self.dvdrop1( self.dvln1(kno_dv) ) )
        kno_dv_proj = torch.tanh(kno_dv_proj)
        unk_dv_proj = self.dvproj1( self.dvdrop1( self.dvln1(unk_dv) ) )
        unk_dv_proj = torch.tanh(unk_dv_proj)
        # dv_proj = [batch, seq_len, 12]

        kno_proj = self.embproj1( self.embdrop1( self.embln1(kno_emb) ) )
        kno_proj = torch.tanh(kno_proj)
        unk_proj = self.embproj1( self.embdrop1( self.embln1(unk_emb) ) )
        unk_proj = torch.tanh(unk_proj)
        # emb proj = [batch, seq_len, 12]

        kno_comb = torch.cat((kno_dv_proj, kno_proj), dim=2)
        unk_comb = torch.cat((unk_dv_proj, unk_proj), dim=2)

        kno_dv_proj = self.tfdense( self.tfdrop( self.tfln(kno_comb) ) )
        kno_dv_proj = F.gelu(kno_dv_proj)  # [batch, seq_len, 12]
        unk_dv_proj = self.tfdense( self.tfdrop( self.tfln(unk_comb) ) )
        unk_dv_proj = F.gelu(unk_dv_proj)  # [batch, seq_len, 12]

        # along seq_len dim
        kno_dv_proj = self.avg(kno_dv_proj, kno_mask)  # [batch, 64]
        unk_dv_proj = self.avg(unk_dv_proj, unk_mask)  # [batch, 64]

        dv_comb = torch.cat((kno_dv_proj, unk_dv_proj), dim=1) # [batch, 128]

        hidden = self.dense2( self.drop2( self.ln2(dv_comb) ) )
        hidden = F.gelu(hidden)  # [batch, 24]

        hidden = self.dense3( self.drop3( self.ln3(hidden) ) )  # [batch, 1]

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

        # self.proj_head = DVProjectionHead()
        # self.proj_head = DVProjectionHead_ActiFirst()
        self.proj_head = DVProjectionHead_EmbActi()

        self.tkz = RobertaTokenizer.from_pretrained("roberta-base")
        self.collator = TokenizerCollate(self.tkz)

        self.lossfunc = nn.BCEWithLogitsLoss()

        self.acc = Accuracy(threshold=0.0)
        self.f1 = F1(threshold=0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["learning_rate"])
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                    num_warmup_steps=10,
                                                                                    num_training_steps=5000,
                                                                                    num_cycles=10)
        schedulers = [    
        {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }]
        return [optimizer], schedulers

    def train_dataloader(self):
        # self.dataset_train = PANDataset('./data_pickle_cutcombo/pan_all_cls/train_kucombo_only.pickle')
        # self.dataset_train = PANDataset('./data_pickle_cutcombo/pan_14e_cls/train_essays.pickle')
        self.dataset_train = PANDataset('./data_pickle_cutcombo/pan_14n_cls/train_novels_kucombo_only.pickle')
        self.loader_train = DataLoader(self.dataset_train,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_train

    def val_dataloader(self):
        # self.dataset_val = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
        self.dataset_val = PANDataset('./data_pickle_cutcombo/pan_14n_cls/test02_novels_onecut.pickle')
        self.loader_val = DataLoader(self.dataset_val,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_val

    def test_dataloader(self):
        # self.dataset_test = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
        self.dataset_test = PANDataset('./data_pickle_cutcombo/pan_14n_cls/test02_novels_onecut.pickle')
        self.loader_test = DataLoader(self.dataset_test,
                                        batch_size=self.train_config["batch_size"],
                                        collate_fn=self.collator,
                                        num_workers=4,
                                        pin_memory=True, drop_last=False, shuffle=False)
        return self.loader_test
    
    @autocast()
    def forward(self, inputs, onedoc_enc=False):
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

        if onedoc_enc:
            doc_ids, doc_mask = inputs
            doc_embed, doc_pred = one_doc_embed(input_ids=doc_ids, input_mask=doc_mask)
            doc_dv = doc_pred - doc_embed
            return doc_pred, doc_embed, doc_dv
        else:
            labels, kno_ids, kno_mask, unk_ids, unk_mask = inputs

            kno_embed, kno_pred = one_doc_embed(input_ids=kno_ids, input_mask=kno_mask)
            unk_embed, unk_pred = one_doc_embed(input_ids=unk_ids, input_mask=unk_mask)

            kno_dv = kno_pred - kno_embed
            unk_dv = unk_pred - unk_embed

            # logits = self.proj_head(kno_dv, kno_mask[:,1:-1], unk_dv, unk_mask[:,1:-1])
            logits = self.proj_head(kno_embed, kno_dv, kno_mask[:,1:-1], 
                                    unk_embed, unk_dv, unk_mask[:,1:-1])

            logits = torch.squeeze(logits)
            labels = labels.float()
            loss = self.lossfunc(logits, labels)

            return (loss, logits, (kno_embed, kno_pred, unk_embed, unk_pred))
    
    def training_step(self, batch, batch_idx):
        labels, kno_ids, kno_mask, unk_ids, unk_mask  = batch
        
        loss, logits, outputs = self( (labels, kno_ids, kno_mask, unk_ids, unk_mask) )
        
        self.log("train_loss", loss)
        self.log("logits mean", logits.mean())
        self.log("LR", self.trainer.optimizers[0].param_groups[0]['lr'] )

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
        self.log('eval F1', self.f1.compute())

# %%
@rank_zero_only
def wandb_save(wandb_logger, train_config):
    wandb_logger.log_hyperparams(train_config)
    wandb_logger.experiment.save('./_dv_pan14e_roberta_projmodel.py', policy="now")

# %%
if __name__ == "__main__":
    train_config = {}
    train_config["cache_dir"] = "./cache/"
    train_config["epochs"] = 14
    train_config["batch_size"] = 256
    # train_config["accumulate_grad_batches"] = 12
    train_config["gradient_clip_val"] = 1.5
    train_config["learning_rate"] = 1e-4

    pl.seed_everything(42)

    wandb_logger = WandbLogger(name='pan14e_emb+dv_24-24-24_tanh',project='AVDV_PAN14N')
    wandb_save(wandb_logger, train_config)

    model = LightningLongformerCLS(train_config)
    # model = LightningLongformerCLS.load_from_checkpoint("AVDV/10lzwg3i/checkpoints/epoch=8-step=1511.ckpt", config=train_config)
    
    cp_valloss = ModelCheckpoint(save_top_k=5, monitor='val_loss', mode='min')
    trainer = pl.Trainer(max_epochs=train_config["epochs"],
                        # accumulate_grad_batches=train_config["accumulate_grad_batches"],
                        gradient_clip_val=train_config["gradient_clip_val"],

                        gpus=[4],
                        num_nodes=1,
                        # accelerator='ddp',

                        amp_backend='native',
                        precision=16,

                        logger=wandb_logger,
                        log_every_n_steps=1,

                        limit_val_batches=40,
                        checkpoint_callback=cp_valloss
                        )

    trainer.fit(model)