#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn.functional as F
# from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)

#%%
import numpy as np
import pandas as pd
from tqdm import tqdm

#%%
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#%%
model = RobertaForMaskedLM.from_pretrained('roberta-base')
_ = model.eval()
_ = model.to('cuda:6')

for param in model.parameters():
    param.requires_grad = False

pred_model = model.roberta
enco_model = pred_model.embeddings.word_embeddings

#%%
text = "Assesing ones strengths and weaknesses in any situation is a hard task." + \
       " Usually we are not so good at recognizing our strengths, instead we spend our time being critical of ourselves," + \
       " thus we tend to present a much more detailed side of our flaws."

encoded_text = tokenizer(text)

input_mask = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')
input_ids = torch.tensor([encoded_text["input_ids"]]).to('cuda:6')

output_encode = enco_model(input_ids)

pred_tensor = []
for i in range(1, len(encoded_text["input_ids"])-1):
    masked_ids = input_ids.clone()
    masked_ids[0,i] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = pred_model(input_ids=masked_ids, attention_mask=input_mask, return_dict=False)[0]
        
#%%
def one_doc_embed(enc_model, pred_model, input_ids, input_mask, mask_n):
    with torch.no_grad():
        embed = enc_model(input_ids)[0,1:-1,:].detach()  # remove start and end symbol

        result_embed = []
        result_pred = []
        for i in range(1, len(encoded_text["input_ids"])-1-mask_n):
            masked_ids = input_ids.clone()
            masked_ids[0,i:(i+mask_n)] = tokenizer.mask_token_id

            output = pred_model(input_ids=masked_ids, attention_mask=input_mask, return_dict=False)[0]
            result_embed.append( embed[i:(i+mask_n),:] )
            result_pred.append(output[0,i:(i+mask_n),:].detach())

        result_embed = torch.cat(result_embed, dim=0)
        result_pred = torch.cat(result_pred, dim=0)
        return {"e": result_embed, "p": result_pred}

# %%
df_test02 = pd.read_pickle('./data_pickle_trfm/pan_14e_cls/test02_essays.pickle')
df_test02

# %%
encoded_text = tokenizer(df_test02.iloc[0,2][:1000])
# tokenizer.convert_ids_to_tokens(encoded_text["input_ids"])

# %%
input_mask = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')
input_ids = torch.tensor([encoded_text["input_ids"]]).to('cuda:6')

result = one_doc_embed( enco_model, pred_model, input_ids, input_mask, mask_n=3 )

# %%
result_list = []
for i in tqdm(range(len(df_test02))):
# for i in tqdm(range(10)):
    encoded_text = tokenizer(df_test02.iloc[i,1][0][:1000])
    input_mask = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')
    input_ids = torch.tensor([encoded_text["input_ids"]]).to('cuda:6')
    k_result = one_doc_embed( enco_model, pred_model, input_ids, input_mask, mask_n=3 )

    encoded_text = tokenizer(df_test02.iloc[i,2][:1000])
    input_mask = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')
    input_ids = torch.tensor([encoded_text["input_ids"]]).to('cuda:6')
    u_result = one_doc_embed( enco_model, pred_model, input_ids, input_mask, mask_n=3 )

    result_list.append({"k": k_result, "u": u_result})

# %%
result = []
for i in range(len(result_list)):
#     print(i)
    result_k = torch.mean(result_list[i]["k"]["e"] - result_list[i]["k"]["p"], dim=0, keepdim=True)
    result_u = torch.mean(result_list[i]["u"]["e"] - result_list[i]["u"]["p"], dim=0, keepdim=True)
    r = F.cosine_similarity(result_k, result_u).cpu().item()
    
    result.append(r)
    
result = np.array(result)
# %%
result

# %%
pred = result > np.median(result)
pred
# %%
# %%
truth = df_test02.iloc[:,0].to_numpy() == "Y"
truth

# %%
np.sum( pred == truth ) / len(truth)
# %%