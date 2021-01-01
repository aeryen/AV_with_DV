# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import torch
from _dv_pan14e_roberta_projmodel import LightningLongformerCLS

# %%

train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["epochs"] = 16
train_config["batch_size"] = 256
# train_config["accumulate_grad_batches"] = 12
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 1e-5

# %%

model = LightningLongformerCLS.load_from_checkpoint("./AVDV/143ui570/checkpoints/epoch=5-step=1979.ckpt", config=train_config)
test_dl = model.test_dataloader()

# %%
model = model.to("cuda:6")
model = model.eval()

# %%
batch = next(iter(test_dl))
# %%
batch = [d.to("cuda:6") for d in batch]

# %%
labels, kno_ids, kno_mask, unk_ids, unk_mask = batch

# %%
with torch.no_grad():
    loss, logits, outputs = model(batch)
# %%
pred = logits.detach().cpu().numpy()
# %%
pred_bool = pred > np.median(pred)
# %%
truth = labels.cpu().numpy()
# %%
np.sum( pred_bool == truth ) / len(truth)
# %%
