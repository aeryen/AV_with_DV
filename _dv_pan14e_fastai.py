# %%
from fastai.text.all import *
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# %%
import torch.nn.functional as F

# %%
torch.cuda.set_device('cuda:0')
defaults.device = "cuda:0"

# %%
df_train = pd.read_pickle('./data_new/pan_14e_cls/train_essays.pickle')
df_test01 = pd.read_pickle('./data_new/pan_14e_cls/test01_essays.pickle')
df_test02 = pd.read_pickle('./data_new/pan_14e_cls/test02_essays.pickle')
df_train
# %%
len(df_train), len(df_test01), len(df_test02)
# %%
# Getting unique documents
uniq_list = []
for dfi, df in enumerate([df_train, df_test01, df_test02]):
    for i in range(len(df)):
        for doc in df.iloc[i, 1]:
            if(doc not in uniq_list):
                uniq_list.append(doc)

        if(df.iloc[i, 2] not in uniq_list):
            uniq_list.append(df.iloc[i, 2])
len(uniq_list)
# %%
uniq_list = pd.DataFrame(uniq_list)
uniq_list
# %%
dl = TextDataLoaders.from_df(uniq_list,
                             path='./data_new/pan_14e_cls/',
                             is_lm=True,
                             valid_pct=0.1,
                             text_col=0)
# %%
dl.show_batch(max_n=5)
# %%
learn = language_model_learner(dl, AWD_LSTM,
                               metrics=[accuracy, Perplexity()],
                               path="./model/lm_pan14e/",
                               wd=0.1)
# %%
# TEXT = "important to set up rules about how much it is allowed to watch xxup tv"
# N_WORDS = 20
# N_SENTENCES = 4
# preds = learn.predict(TEXT, N_WORDS, temperature=0.75)
# print(preds)

# %%

def one_doc_embed(doc_enc, pred_enc, doc):
    ids = dl.test_dl([doc]).items[0].to(dl.device)

    with torch.no_grad():
        pred_enc.reset()
        embed = doc_enc(ids[None])[0,:,:].detach()
        pred = pred_enc(ids[None])[0,:,:].detach()

        return {"e": embed, "p": pred}


def pred_raw(learn, df):
    model = learn.model
    model = model.to(device=dl.device)
    model.eval()
    doc_enc = model[0].encoder
    pred_enc = model[0]

    prob_list = []
    label_list = []

    for i in tqdm(range(len(df))):
        k_results = []
        for k_doc in df.iloc[i, 1]:
            k_results.append(one_doc_embed(doc_enc, pred_enc, k_doc))

        u_results = one_doc_embed(doc_enc, pred_enc, df.iloc[i, 2])
        prob_list.append({"k": k_results, "u": u_results})
        label_list.append(df.iloc[i, 0])

    return prob_list, label_list


# %%
test_prob, test_label = pred_raw(learn, df_test02)

# %%
result = dl.test_dl([df.iloc[0, 2]])

# %%
# Experiment with RAW Values

# %%
truth = np.array(test_label) == "Y"
truth
# %%
result = []
for i in range(len(test_prob)):
#     print(i)
    result_k = torch.mean(test_prob[i]["k"][0]["e"][1:,:] - test_prob[i]["k"][0]["p"][-1,:], dim=0, keepdim=True)
    result_u = torch.mean(test_prob[i]["u"]["e"][1:,:] - test_prob[i]["u"]["p"][-1,:], dim=0, keepdim=True)
    r = F.cosine_similarity(result_k, result_u).cpu().item()
    
    result.append(r)
    
result = np.array(result)
# %%
pred = result > np.median(result)
pred
# %%
np.sum( pred == truth ) / len(truth)
# %%
