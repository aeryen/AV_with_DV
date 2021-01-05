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
from _dv_roberta_projmodel_KUEP import LightningLongformerCLS, PANDatasetKUEP, TokenizerCollateKUEP

from sklearn.metrics import roc_auc_score

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
def eval_model_onecut(model, datalist):
    model = model.to(train_config["gpu_id"])
    model = model.eval()

    truth = []
    outputs = []
    with torch.no_grad():
        for data in tqdm(datalist):
            label = data["l"]
            kno = data["k"]
            unk = data["u"]
            kno_mask = kno["input_mask"][0:1,1:-1].to(train_config["gpu_id"])
            unk_mask = unk["input_mask"][0:1,1:-1].to(train_config["gpu_id"])
            kno_pred = kno["p"][0:1,:,:].to(train_config["gpu_id"])
            kno_emb = kno["e"][0:1,:,:].to(train_config["gpu_id"])
            unk_pred = unk["p"][0:1,:,:].to(train_config["gpu_id"])
            unk_emb = unk["e"][0:1,:,:].to(train_config["gpu_id"])
            kno_dv = kno_pred - kno_emb
            unk_dv = unk_pred - unk_emb
            
            logits = model.proj_head(kno_emb, kno_dv, kno_mask, 
                                     unk_emb, unk_dv, unk_mask )

            truth.append(label)
            outputs.append(logits.item())

    truth = np.array(truth)
    outputs = np.array(outputs)

    print("\n\n")    
    pred = (outputs > np.median(outputs) )
    acc = np.sum( pred == truth ) / len(truth)
    print(acc)
    pred = (outputs > 0)
    acc = np.sum( pred == truth ) / len(truth)
    print(acc)

    return outputs, truth


def eval_model_wholedoc_EMBDV(model, data_list):
    result = []
    model = model.to(train_config["gpu_id"])
    model = model.eval()
    labels = []
    with torch.no_grad():
        for data in tqdm(data_list):
            label = data["l"]
            kno = data["k"]
            unk = data["u"]
            kno_mask = kno["input_mask"].to(train_config["gpu_id"])
            unk_mask = unk["input_mask"].to(train_config["gpu_id"])
            kno_pred = kno["p"].to(train_config["gpu_id"])
            kno_emb = kno["e"].to(train_config["gpu_id"])
            unk_pred = unk["p"].to(train_config["gpu_id"])
            unk_emb = unk["e"].to(train_config["gpu_id"])
            kno_dv = kno_pred - kno_emb
            unk_dv = unk_pred - unk_emb

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
            kno_dv_proj = torch.tanh(kno_dv_proj)  # [batch, seq_len, 12]
            unk_dv_proj = model.proj_head.tfdense( model.proj_head.tfdrop( model.proj_head.tfln(unk_comb) ) )
            unk_dv_proj = torch.tanh(unk_dv_proj)  # [batch, seq_len, 12]

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
            labels.append( label )

    print("\n\n")
    truth = np.array(labels)
    result = np.array(result)
    pred_med = result > np.median(result)
    acc = np.sum( pred_med == truth ) / len(truth)
    print(acc)
    pred_zero = result > 0
    acc = np.sum( pred_zero == truth ) / len(truth)
    print(acc)
    return result, truth, pred_med, pred_zero, acc

# %%
def calc_cat1(answers, truth):
    # (1/n)*(nc+(nu*nc/n))
    n_correct = 0
    n_undecided = 0
    n = len(answers)
    for k, v in enumerate(answers):
        if v == 0.5:
            n_undecided += 1
        else:
            n_correct += (v > 0.5) == truth[k]

    scale = 1.0 / n
    return (n_correct + n_undecided * n_correct * scale) * scale

# %%
# PAN14N
# emb+dv 12 24 24 tanh
# model = LightningLongformerCLS.load_from_checkpoint("AVDV/2npel9bz/checkpoints/epoch=7-step=2639.ckpt", config=train_config)
# pan14e best2
# model = LightningLongformerCLS.load_from_checkpoint("AVDV/3jzkwqb5/checkpoints/epoch=0-step=963.ckpt", config=train_config)

# PAN14N
model = LightningLongformerCLS.load_from_checkpoint("AVDV_PAN14N/16erabgu/checkpoints/epoch=10-step=3662.ckpt", config=train_config)

# %%
datalist = torch.load("data_pickle_cutcombo/pan_14n_cls/test02_KUEP.pt")

# %%
datalist = torch.load("data_pickle_cutcombo/pan_14e_cls/test02_essays_KUEP.pt")

# %%
logits, truth = eval_model_onecut(model, datalist)

# %%
result, truth, pred_med, pred_zero, acc = eval_model_wholedoc_EMBDV(model, datalist)

# %%
roc_result = roc_auc_score(truth, result)

# %%
cat_result = calc_cat1(pred_med.astype(np.int), truth.astype(np.int))

# %%
roc_result * cat_result

# %%
