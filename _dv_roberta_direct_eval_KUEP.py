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

from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_auc_score

# %%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 16
train_config["batch_size"] = 256
# train_config["accumulate_grad_batches"] = 12
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 1e-5
train_config["gpu_id"] = "cuda:4"

# %%
def average(embedding: torch.Tensor, mask: torch.Tensor):
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

def eval_dvdist(data_list, threshold=0):
    result = []
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


            kno_dv_doc = []
            kno_mask_doc = []
            for i in range(kno_dv.shape[0]):
                kno_dv_doc.append( kno_dv[i:i+1,:,:] )
                kno_mask_doc.append( kno_mask[i:i+1,1:-1] )
            kno_dv_doc = torch.cat(kno_dv_doc, dim=1)
            kno_mask_doc = torch.cat(kno_mask_doc, dim=1)

            unk_dv_doc = []
            unk_mask_doc = []
            for i in range(unk_dv.shape[0]):
                unk_dv_doc.append( unk_dv[i:i+1,:,:] )
                unk_mask_doc.append( unk_mask[i:i+1,1:-1] )
            unk_dv_doc = torch.cat(unk_dv_doc, dim=1)
            unk_mask_doc = torch.cat(unk_mask_doc, dim=1)

            kno_dv_proj = average(kno_dv_doc, kno_mask_doc)
            unk_dv_proj = average(unk_dv_doc, unk_mask_doc)

            dvdist = cosine_similarity( kno_dv_proj, unk_dv_proj )

            result.append( dvdist.item() )
            labels.append( label )

    print("\n\n")
    truth = np.array(labels)
    result = np.array(result)
    pred_med = result > np.median(result)
    acc = np.sum( pred_med == truth ) / len(truth)
    print(acc)
    pred_zero = result > threshold
    acc = np.sum( pred_zero == truth ) / len(truth)
    print(acc)
    return result, truth, pred_med, pred_zero

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
datalist = torch.load("data_pickle_cutcombo/pan_13_cls/test02_KUEP.pt")

# %%
datalist_train = torch.load("data_pickle_cutcombo/pan_14n_cls/train_KUEP.pt")
datalist_test = torch.load("data_pickle_cutcombo/pan_14n_cls/test02_KUEP.pt")

# %%
datalist = torch.load("data_pickle_cutcombo/pan_14e_cls/test02_essays_KUEP.pt")

# %%
datalist = torch.load("data_pickle_cutcombo/pan_15_cls/test_KUEP.pt")

# %%
result, truth, pred_med, pred_zero = eval_dvdist(datalist_train)

# %%
m = np.median(result)

# %%
result, truth, pred_med, pred_zero = eval_dvdist(datalist_test, threshold=m)

# %%
result, truth, pred_med, pred_zero = eval_dvdist(datalist)

# %%
roc_result = roc_auc_score(truth, result)
roc_result

# %%
cat_result = calc_cat1(pred_med.astype(np.int), truth.astype(np.int))
cat_result

# %%
roc_result * cat_result

# %%
