# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
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
train_config["gpu_id"] = "cuda:5"

# %%
def eval_model_onecut(model, test_dl):
    model = model.to("cuda:5")
    model = model.eval()
    batch = next(iter(test_dl))
    batch = [d.to("cuda:5") for d in batch]
    with torch.no_grad():
        loss, logits, outputs = model(batch)

    pred = (logits > logits.median()).cpu().numpy()
    truth = batch[0].cpu().numpy()

    acc = np.sum( pred == truth ) / len(truth)
    print(acc)
    return acc

# %%
# proj12_nn12-12-1tanh_good
model = LightningLongformerCLS.load_from_checkpoint("AVDV/239gfgeh/checkpoints/epoch=4-step=1649.ckpt", config=train_config)

# %%
tkz = RobertaTokenizer.from_pretrained("roberta-base")

# %%
dataset_14e = PANDataset('./data_pickle_cutcombo/pan_14e_cls/test02_essays_onecut.pickle')
dl_14e = DataLoader(dataset_14e,
                        batch_size=256,
                        collate_fn=TokenizerCollate(tkz),
                        num_workers=0, drop_last=False, shuffle=False)
acc = eval_model_onecut(model, dl_14e)

# %%
dataset_14n = PANDataset('./data_pickle_cutcombo/pan_14n_cls/test02_novels_onecut.pickle')
dl_14n = DataLoader(dataset_14n,
                        batch_size=256,
                        collate_fn=TokenizerCollate(tkz),
                        num_workers=0, drop_last=False, shuffle=False)
acc = eval_model_onecut(model, dl_14n)

# %%
def eval_model_wholedoc(model, test_dl):
    model = model.to("cuda:5")
    model = model.eval()
    batch = next(iter(test_dl))
    batch = [d.to("cuda:5") for d in batch]
    with torch.no_grad():
        loss, logits, outputs = model(batch)

    pred = (logits > logits.median()).cpu().numpy()
    truth = batch[0].cpu().numpy()

    acc = np.sum( pred == truth ) / len(truth)
    print(acc)
    return acc

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
result = []
with torch.no_grad():
    for batch in tqdm(dl_14e):
        batch = [d.to("cuda:5") for d in batch]
        kno_ids, kno_mask, unk_ids, unk_mask = batch
        model = model.to("cuda:5")
        model = model.eval()
        
        kno_dv = model( (kno_ids, kno_mask) , onedoc_enc=True)
        unk_dv = model( (unk_ids, unk_mask) , onedoc_enc=True)

        # logits = model.proj_head(kno_dv, kno_mask[:,1:-1], unk_dv, unk_mask[:,1:-1])
        kno_dv_proj = model.proj_head.proj1( model.proj_head.drop1( model.proj_head.ln1(kno_dv) ) )
        unk_dv_proj = model.proj_head.proj1( model.proj_head.drop1( model.proj_head.ln1(unk_dv) ) )

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
        dv_comb = torch.tanh(dv_comb)

        hidden = model.proj_head.drop2( model.proj_head.ln2(dv_comb) )
        hidden = model.proj_head.dense2(hidden)
        hidden = torch.tanh(hidden)  # [batch, 24]

        hidden = model.proj_head.drop3( model.proj_head.ln3(hidden) )
        hidden = model.proj_head.dense3(hidden)  # [batch, 1]

        result.append( hidden.item() )


# %%
