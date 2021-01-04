# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import RobertaTokenizer
from _dv_pan14e_roberta_projmodel import LightningLongformerCLS, PANDataset, TokenizerCollate

# %%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 16
train_config["batch_size"] = 256
# train_config["accumulate_grad_batches"] = 12
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 1e-5
train_config["gpu_id"] = "cuda:0"

# %%
def eval_model_onecut(model, test_dl):
    model = model.to(train_config["gpu_id"])
    model = model.eval()
    batch = next(iter(test_dl))
    batch = [d.to(train_config["gpu_id"]) for d in batch]
    with torch.no_grad():
        loss, logits, outputs = model(batch)

    pred = (logits > logits.median()).cpu().numpy()
    truth = batch[0].cpu().numpy()

    acc = np.sum( pred == truth ) / len(truth)
    print(acc)

    return acc, logits.cpu().numpy(), truth


# %%
# proj12_nn12-12-1tanh_good
# model = LightningLongformerCLS.load_from_checkpoint("AVDV/239gfgeh/checkpoints/epoch=4-step=1649.ckpt", config=train_config)

# good emb+dv
# model = LightningLongformerCLS.load_from_checkpoint("AVDV/3tbf9ol3/checkpoints/epoch=0-step=329.ckpt", config=train_config)

# good acti
model = LightningLongformerCLS.load_from_checkpoint("AVDV/26liqdv3/checkpoints/epoch=3-step=1319.ckpt", config=train_config)


# %%
tkz = RobertaTokenizer.from_pretrained("roberta-base")

# %%
dataset_14e = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
dl_14e = DataLoader(dataset_14e,
                        batch_size=256,
                        collate_fn=TokenizerCollate(tkz),
                        num_workers=0, drop_last=False, shuffle=False)
acc, logits, truth = eval_model_onecut(model, dl_14e)

# %%
dataset_14n = PANDataset('./data_pickle_cutcombo/pan_14n_cls/test02_novels_onecut.pickle')
dl_14n = DataLoader(dataset_14n,
                        batch_size=256,
                        collate_fn=TokenizerCollate(tkz),
                        num_workers=0, drop_last=False, shuffle=False)
acc = eval_model_onecut(model, dl_14n)

# %%
def eval_model_wholedoc_ACTI(model, test_dl):
    result = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            batch = [d.to(train_config["gpu_id"]) for d in batch]
            kno_ids, kno_mask, unk_ids, unk_mask = batch
            model = model.to(train_config["gpu_id"])
            model = model.eval()
            
            kno_pred, kno_emb, kno_dv = model( (kno_ids, kno_mask) , onedoc_enc=True)
            unk_pred, unk_emb, unk_dv = model( (unk_ids, unk_mask) , onedoc_enc=True)

            kno_dv_proj = model.proj_head.proj1( model.proj_head.drop1( model.proj_head.ln1(kno_dv) ) )
            kno_dv_proj = F.gelu(kno_dv_proj)
            unk_dv_proj = model.proj_head.proj1( model.proj_head.drop1( model.proj_head.ln1(unk_dv) ) )
            unk_dv_proj = F.gelu(unk_dv_proj)
            # dv_proj = [batch, seq_len, 12]

            kno_dv_proj = model.proj_head.densep( model.proj_head.dropp( model.proj_head.lnp(kno_dv_proj) ) )
            kno_dv_proj = F.gelu(kno_dv_proj)  # [batch, seq_len, 12]
            unk_dv_proj = model.proj_head.densep( model.proj_head.dropp( model.proj_head.lnp(unk_dv_proj) ) )
            unk_dv_proj = F.gelu(unk_dv_proj)  # [batch, seq_len, 12]

            kno_dv_doc = []
            kno_mask_doc = []
            for i in range(kno_dv_proj.shape[0]):
                kno_dv_doc.append( kno_dv_proj[i:i+1,:,:] )
                kno_mask_doc.append( kno_mask[i:i+1,1:-1] )
            kno_dv_doc = torch.cat(kno_dv_doc, dim=1)
            kno_mask_doc = torch.cat(kno_mask_doc, dim=1)

            unk_dv_doc = []
            unk_mask_doc = []
            for i in range(unk_dv_proj.shape[0]):
                unk_dv_doc.append( unk_dv_proj[i:i+1,:,:] )
                unk_mask_doc.append( unk_mask[i:i+1,1:-1] )
            unk_dv_doc = torch.cat(unk_dv_doc, dim=1)
            unk_mask_doc = torch.cat(unk_mask_doc, dim=1)

            kno_dv_proj = model.proj_head.avg(kno_dv_doc, kno_mask_doc)
            unk_dv_proj = model.proj_head.avg(unk_dv_doc, unk_mask_doc)

            dv_comb = torch.cat((kno_dv_proj, unk_dv_proj), dim=1) # [batch, 24]

            hidden = model.proj_head.dense2( model.proj_head.drop2( model.proj_head.ln2(dv_comb) ) )
            hidden = torch.tanh(hidden)  # [batch, 24]

            hidden = model.proj_head.dense3 (model.proj_head.drop3( model.proj_head.ln3(hidden) ) )  # [batch, 1]

            result.append( hidden.item() )

def eval_model_wholedoc_EMBDV(model, test_dl):
    result = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            batch = [d.to(train_config["gpu_id"]) for d in batch]
            kno_ids, kno_mask, unk_ids, unk_mask = batch
            model = model.to(train_config["gpu_id"])
            model = model.eval()
            
            kno_pred, kno_emb, kno_dv = model( (kno_ids, kno_mask) , onedoc_enc=True)
            unk_pred, unk_emb, unk_dv = model( (unk_ids, unk_mask) , onedoc_enc=True)

            kno_dv_proj = model.proj_head.dvproj1( model.proj_head.dvdrop1( model.proj_head.dvln1(kno_dv) ) )
            kno_dv_proj = torch.tanh(kno_dv_proj)
            unk_dv_proj = model.proj_head.dvproj1( model.proj_head.dvdrop1( model.proj_head.dvln1(unk_dv) ) )
            unk_dv_proj = torch.tanh(unk_dv_proj)

            kno_proj = model.proj_head.embproj1( model.proj_head.embdrop1( model.proj_head.embln1(kno_emb) ) )
            kno_proj = torch.tanh(kno_proj)
            unk_proj = model.proj_head.embproj1( model.proj_head.embdrop1( model.proj_head.embln1(unk_emb) ) )
            unk_proj = torch.tanh(unk_proj)

            kno_comb = torch.cat((kno_dv_proj, kno_proj), dim=2)
            unk_comb = torch.cat((unk_dv_proj, unk_proj), dim=2)

            kno_dv_proj = model.proj_head.tfdense( model.proj_head.tfdrop( model.proj_head.tfln(kno_comb) ) )
            kno_dv_proj = F.gelu(kno_dv_proj)  # [batch, seq_len, 12]
            unk_dv_proj = model.proj_head.tfdense( model.proj_head.tfdrop( model.proj_head.tfln(unk_comb) ) )
            unk_dv_proj = F.gelu(unk_dv_proj)  # [batch, seq_len, 12]

            kno_dv_doc = []
            kno_mask_doc = []
            for i in range(kno_dv_proj.shape[0]):
                kno_dv_doc.append( kno_dv_proj[i:i+1,:,:] )
                kno_mask_doc.append( kno_mask[i:i+1,1:-1] )
            kno_dv_doc = torch.cat(kno_dv_doc, dim=1)
            kno_mask_doc = torch.cat(kno_mask_doc, dim=1)

            unk_dv_doc = []
            unk_mask_doc = []
            for i in range(unk_dv_proj.shape[0]):
                unk_dv_doc.append( unk_dv_proj[i:i+1,:,:] )
                unk_mask_doc.append( unk_mask[i:i+1,1:-1] )
            unk_dv_doc = torch.cat(unk_dv_doc, dim=1)
            unk_mask_doc = torch.cat(unk_mask_doc, dim=1)

            kno_dv_proj = model.proj_head.avg(kno_dv_doc, kno_mask_doc)
            unk_dv_proj = model.proj_head.avg(unk_dv_doc, unk_mask_doc)

            dv_comb = torch.cat((kno_dv_proj, unk_dv_proj), dim=1) # [batch, 24]

            hidden = model.proj_head.dense2( model.proj_head.drop2( model.proj_head.ln2(dv_comb) ) )
            hidden = torch.tanh(hidden)  # [batch, 24]

            hidden = model.proj_head.dense3 (model.proj_head.drop3( model.proj_head.ln3(hidden) ) )  # [batch, 1]

            result.append( hidden.item() )

    pred = np.array(result)
    pred_bol = pred > np.median(pred)
    truth = ( dataset.df.iloc[:,0] == "Y" ).to_numpy()
    acc = np.sum( pred_bol == truth ) / len(truth)
    print(acc)
    return pred, acc


def eval_model_cutdist(model, test_dl):
    result = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            batch = [d.to(train_config["gpu_id"]) for d in batch]
            kno_ids, kno_mask, unk_ids, unk_mask = batch
            model = model.to(train_config["gpu_id"])
            model = model.eval()
            
            kno_pred, kno_embed, kno_dv = model( (kno_ids, kno_mask) , onedoc_enc=True)
            unk_pred, unk_embed, unk_dv = model( (unk_ids, unk_mask) , onedoc_enc=True)

            dist_val = []
            # print(kno_dv.shape[0])
            # print(unk_dv.shape[0])
            for i in range(kno_dv.shape[0]):
                for j in range(unk_dv.shape[0]):
                    logit = model.proj_head( kno_embed[i:i+1,:,:], kno_dv[i:i+1,:,:], kno_mask[i:i+1,1:-1],
                                             unk_embed[j:j+1,:,:], unk_dv[j:j+1,:,:], unk_mask[j:j+1,1:-1] )
                    dist_val.append(logit.item())
            
            result.append(dist_val)
    return result

class OneDocCollate:
    def __init__(self, tkz):
        self.tkz = tkz
    
    def __call__(self, docs):
        label, known, unknown = docs[0]
        encode_kno = self.tkz(known, truncation=True, padding="max_length", max_length=128)
        encode_unk = self.tkz(unknown, truncation=True, padding="max_length", max_length=128)
        return torch.tensor(encode_kno["input_ids"], dtype=torch.int64), \
                torch.tensor(encode_kno["attention_mask"], dtype=torch.int64), \
                torch.tensor(encode_unk["input_ids"], dtype=torch.int64), \
                torch.tensor(encode_unk["attention_mask"], dtype=torch.int64)

# %%
dataset = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_cutlist.pickle')
dl_14e = DataLoader(dataset,
                        batch_size=1,
                        collate_fn=OneDocCollate(tkz),
                        num_workers=0, drop_last=False, shuffle=False)

# %%
result = eval_model_cutdist(model, dl_14e)

# %%
result[0]

# %%
import seaborn as sns

# %%
pos_dist = []
neg_dist = []
for i in range( len( dataset.df ) ):
    if dataset.df.iloc[i,0] == "Y":
        pos_dist.append( np.mean(result[i]) )
    else:
        neg_dist.append( np.mean(result[i]) )

# %%
fig = sns.distplot(pos_dist, kde=False, rug=True, hist=True, bins=20,  kde_kws={"color": "blue"} )
fig = sns.distplot(neg_dist, kde=False, rug=True, hist=True, bins=20,  kde_kws={"color": "red"} )

# %%
dist_val = []
for i in range( len( dataset.df ) ):
    dist_val.append( np.max(result[i]) )

dist_val = np.array(dist_val)
pred = dist_val > np.max( dist_val )

truth = ( dataset.df.iloc[:,0] == "Y" ).to_numpy()
acc = np.sum( pred == truth ) / len(truth)
print(acc)
# %%

# %%
pred, acc = eval_model_wholedoc_EMBDV(model, dl_14e)

# %%
acc

# %%
