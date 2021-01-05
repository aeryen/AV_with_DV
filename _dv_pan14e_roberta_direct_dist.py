# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn.functional as F
# from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
train_config = {}
train_config["cache_dir"] = "./cache/"
train_config["gpu_id"] = "cuda:2"

# %%
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# %%
model = RobertaForMaskedLM.from_pretrained('roberta-base')
_ = model.eval()
_ = model.to(train_config["gpu_id"])

for param in model.parameters():
    param.requires_grad = False

pred_model = model.roberta
enco_model = pred_model.embeddings.word_embeddings

# %%
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
        
# %%
def one_doc_embed(enc_model, pred_model, input_ids, input_mask, mask_n=1):
    with torch.no_grad():
        uniq_mask = []
        uniq_input, inverse_indices = torch.unique( input_ids, return_inverse=True, dim=0 )
        invi = inverse_indices.detach().cpu().numpy()
        for i in range( uniq_input.shape[0] ):
            first_index = np.where(invi == i)[0][0]
            uniq_mask.append(input_mask[first_index,:])

        input_ids = uniq_input
        input_mask = torch.stack(uniq_mask, dim=0)

        embed = enc_model(input_ids)

        result_embed = []
        result_pred = []
        # skip start and end symbol
        masked_ids = input_ids.clone()
        for i in range(1, input_ids.shape[1]-mask_n):
            masked_ids[:,i:(i+mask_n)] = tokenizer.mask_token_id

            output = pred_model(input_ids=masked_ids, attention_mask=input_mask, return_dict=False)[0]
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
    return {"e": rec_embed.detach().cpu(), "p": rec_pred.detach().cpu()}

# %%
df_test02 = pd.read_pickle('./data_pickle_cutcombo/pan_14n_cls/test02_novels_cutlist.pickle')
df_test02

# %%
result_list = []
for i in tqdm(range(109, len(df_test02))):
# for i in tqdm(range(10)):
    encoded_text = tokenizer(df_test02.iloc[i,1], truncation=True, padding="max_length", max_length=128)
    input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64).to(train_config["gpu_id"])
    input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64).to(train_config["gpu_id"])
    k_result = one_doc_embed( enco_model, pred_model, input_ids, input_mask, mask_n=1 )

    encoded_text = tokenizer(df_test02.iloc[i,2], truncation=True, padding="max_length", max_length=128)
    input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64).to(train_config["gpu_id"])
    input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64).to(train_config["gpu_id"])
    u_result = one_doc_embed( enco_model, pred_model, input_ids, input_mask, mask_n=1 )

    l = df_test02.iloc[i,0]=="Y"
    result_list.append({"l": l, "k": k_result, "u": u_result})

# %%
for item in result_list:
    item["k"]["e"] = item["k"]["e"].detach().cpu()
    item["k"]["p"] = item["k"]["p"].detach().cpu()
    item["u"]["e"] = item["u"]["e"].detach().cpu()
    item["u"]["p"] = item["u"]["p"].detach().cpu()

# %%
torch.save( result_list, "./data_pickle_cutcombo/pan_14n_cls/test02_temp2_KUEP.pt" )

# %%
result_list1 = torch.load("./data_pickle_cutcombo/pan_14n_cls/test02_temp_KUEP.pt" )
result_list2 = torch.load("./data_pickle_cutcombo/pan_14n_cls/test02_temp2_KUEP.pt" )

# %%
result_all = result_list1
result_all.extend( result_list2 )

# %%
result_new = []
for i in tqdm(range(len(result_all))):
    encoded_text = tokenizer(df_test02.iloc[i,1], truncation=True, padding="max_length", max_length=128)
    input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64)
    input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64)
    k_result = {}
    k_result["e"] = result_all[i]["k"]["e"].cpu()
    k_result["p"] = result_all[i]["k"]["p"].cpu()
    k_result["input_ids"] = input_ids.cpu()
    k_result["input_mask"] = input_mask.cpu()

    encoded_text = tokenizer(df_test02.iloc[i,2], truncation=True, padding="max_length", max_length=128)
    input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64)
    input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64)
    u_result = {}
    u_result["e"] = result_all[i]["u"]["e"].cpu()
    u_result["p"] = result_all[i]["u"]["p"].cpu()
    u_result["input_ids"] = input_ids.cpu()
    u_result["input_mask"] = input_mask.cpu()

    l = result_all[i]["l"]
    result_new.append({"l": l, "k": k_result, "u": u_result})

# %%
torch.save( result_new, "./data_pickle_cutcombo/pan_14n_cls/test02_KUEP.pt" )

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
truth = df_test02.iloc[:,0].to_numpy() == "Y"
truth

# %%
np.sum( pred == truth ) / len(truth)
# %%
