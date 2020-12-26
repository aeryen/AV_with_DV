#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
# from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)

#%%
import pandas as pd

#%%
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#%%
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
_ = model.eval()
_ = model.to('cuda:6')

# %%
pred_model = model.roberta
enco_model = pred_model.embeddings.word_embeddings

#%%
text = "Assesing ones strengths and weaknesses in any situation is a hard task." + \
       " Usually we are not so good at recognizing our strengths, instead we spend our time being critical of ourselves," + \
       " thus we tend to present a much more detailed side of our flaws."

encoded_text = tokenizer(text)

# %%
input_mask = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')
input_ids = torch.tensor([encoded_text["input_ids"]]).to('cuda:6')

output_encode = enco_model(input_ids)

#%%
pred_tensor = []
for i in range(1, len(encoded_text["input_ids"])-1):
    masked_ids = input_ids.clone()
    masked_ids[0,i] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = pred_model(input_ids=masked_ids, attention_mask=input_mask, return_dict=False)[0]
        
# %%
df_test02 = pd.read_pickle('./data_new/pan_14e_cls/test02_essays.pickle')
df_test02

#%%