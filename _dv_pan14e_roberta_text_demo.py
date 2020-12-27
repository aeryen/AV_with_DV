#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
# from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)

#%%
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#%%
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
_ = model.eval()
_ = model.to('cuda:6')

#%%
text = "Assesing ones strengths and weaknesses in any situation is a hard task." + \
       " Usually we are not so good at recognizing our strengths, instead we spend our time being critical of ourselves," + \
       " thus we tend to present a much more detailed side of our flaws."

encoded_text = tokenizer(text)

#%%
# TEXT COMPARISON TEST
mask_tensors = torch.tensor([encoded_text["attention_mask"]]).to('cuda:6')

recov_text = tokenizer.convert_ids_to_tokens(encoded_text["input_ids"][1:-1])
print( " ".join(recov_text) )

pred_text = []

for i in range(1, len(encoded_text["input_ids"])-1):

    masked_text = encoded_text["input_ids"].copy()
    masked_text[i] = tokenizer.mask_token_id
    tokens_tensor = torch.tensor([masked_text]).to('cuda:6')
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, attention_mask=mask_tensors)
        predictions = outputs[0]
        predicted_index = torch.argmax(predictions[0, i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        pred_text.append( predicted_token )

print( " ".join(pred_text) )

        
# %%
