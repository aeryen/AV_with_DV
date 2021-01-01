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
train_config["batch_size"] = 200
# train_config["accumulate_grad_batches"] = 12
train_config["gradient_clip_val"] = 1.5
train_config["learning_rate"] = 1e-5

# %%
# proj12_nn12-12-1tanh_good
model = LightningLongformerCLS.load_from_checkpoint("AVDV/239gfgeh/checkpoints/epoch=4-step=1649.ckpt", config=train_config)
test_dl = model.val_dataloader()

# %%
model = model.to("cuda:5")
model = model.eval()

# %%
batch = next(iter(test_dl))

# %%
batch = [d.to("cuda:5") for d in batch]

# %%
result = model.validation_step(batch, 0)

# %%
with torch.no_grad():
    loss, logits, outputs = model(batch)

# %%
pred = logits.detach().cpu().numpy()
# %%
pred_bool = pred > np.median(pred)
# %%
truth = batch[0].cpu().numpy()
# %%
np.sum( pred_bool == truth ) / len(truth)
# %%
def eval_model_onecut(model, ):
