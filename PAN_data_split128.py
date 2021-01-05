# %%
from random import random
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from PAN_data_converter import PANData

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForMaskedLM

# %%
def split_doc(doc, seq_word=100):
    space_map = []

    result = []
    for i,c in enumerate(doc):
        if c.isspace():
            space_map.append(i)
    if space_map[-1] < (len(doc)-1):
        space_map.append( len(doc)-1 )

    last_i = None
    for space_i in range(seq_word,len(space_map),seq_word):
        end_index = min( space_i+5, len(space_map)-1 )
        if space_i == seq_word:
            result.append( doc[ 0:space_map[end_index] ] )
        elif space_i > seq_word:
            result.append( doc[ space_map[last_i-5]:space_map[end_index] ] )
        last_i = end_index
        
    if len(space_map) - last_i > 30: # more than 30 tokens left?
        result.append( doc[ space_map[last_i-5]:space_map[-1] ] )
    return result

# %%
def make_all_combo(df):
    label_list = []
    known_list = []
    unknown_list = []
    prob_index = []
    pos_kno = 0
    pos_unk = 0
    neg_kno = 0
    neg_unk = 0
    for irow in range(len(df)):
        l = df.iloc[irow, 0]
        kno = "\n".join( df.iloc[irow, 1] )
        unk = df.iloc[irow, 2]

        kno_cut = split_doc(kno, seq_word=100)
        unk_cut = split_doc(unk, seq_word=100)
        if l == "Y":
            pos_kno += len(kno_cut)
            pos_unk += len(unk_cut)
        else:
            neg_kno += len(kno_cut)
            neg_unk += len(unk_cut)

        # all known-unknown combo
        for i in range(len(kno_cut)):
            for j in range(len(unk_cut)):
                known_list.append(kno_cut[i])
                unknown_list.append(unk_cut[j])
                label_list.append(l)
                prob_index.append(irow)
        
    result_df = pd.DataFrame({"label":label_list, "known":known_list, "unknown":unknown_list, "prob_index":prob_index})
    print( "pos_kno", pos_kno )
    print( "pos_unk", pos_unk )
    print( "neg_kno", neg_kno )
    print( "neg_unk", neg_unk )
    return result_df

def make_all_combo_KUEP(inlist):
    pos_kno = 0
    pos_unk = 0
    neg_kno = 0
    neg_unk = 0
    result_list = []
    for irow in range(len(inlist)):
        prob = inlist[irow]
        l = prob["l"]
        kno = prob["k"]
        unk = prob["u"]
        if l == True:
            pos_kno += kno["input_ids"].shape[0]
            pos_unk += unk["input_ids"].shape[0]
        else:
            neg_kno += kno["input_ids"].shape[0]
            neg_unk += unk["input_ids"].shape[0]

        for i in range(kno["input_ids"].shape[0]-1):
            for j in range(i+1, kno["input_ids"].shape[0]):
                if random() <= 0.25:
                    k_item = { "input_ids":kno["input_ids"][i,:], "input_mask":kno["input_mask"][i,:],
                           "e":kno["e"][i,:,:], "p":kno["p"][i,:,:] }
                    u_item = { "input_ids":kno["input_ids"][j,:], "input_mask":kno["input_mask"][j,:],
                            "e":kno["e"][j,:,:], "p":kno["p"][j,:,:] }
                    result_list.append( {"l":True, "k":k_item, "u":u_item, "prob_index":irow} )
                if random() <= 0.25:
                    k_item = { "input_ids":kno["input_ids"][i,:], "input_mask":kno["input_mask"][i,:],
                           "e":kno["e"][i,:,:], "p":kno["p"][i,:,:] }
                    u_item = { "input_ids":kno["input_ids"][j,:], "input_mask":kno["input_mask"][j,:],
                            "e":kno["e"][j,:,:], "p":kno["p"][j,:,:] }
                    result_list.append( {"l":True, "k":u_item, "u":k_item, "prob_index":irow} )

        # all known-unknown combo
        for i in range(kno["input_ids"].shape[0]):
            for j in range(unk["input_ids"].shape[0]):
                k_item = { "input_ids":kno["input_ids"][i,:], "input_mask":kno["input_mask"][i,:],
                           "e":kno["e"][i,:,:], "p":kno["p"][i,:,:] }
                u_item = { "input_ids":unk["input_ids"][j,:], "input_mask":unk["input_mask"][j,:],
                           "e":unk["e"][j,:,:], "p":unk["p"][j,:,:] }
                result_list.append( {"l":l, "k":k_item, "u":u_item, "prob_index":irow} )

                result_list.append( {"l":l, "k":u_item, "u":k_item, "prob_index":irow} )
        
    print( "pos_kno", pos_kno )
    print( "pos_unk", pos_unk )
    print( "neg_kno", neg_kno )
    print( "neg_unk", neg_unk )
    return result_list


def make_doc_cut_list(df):
    label_list = []
    known_list = []
    unknown_list = []
    prob_index = []

    for irow in range(len(df)):
        l = df.iloc[irow, 0]
        kno = "\n".join( df.iloc[irow, 1] )
        unk = df.iloc[irow, 2]

        kno_cut = split_doc(kno, seq_word=100)
        unk_cut = split_doc(unk, seq_word=100)

        known_list.append(kno_cut)
        unknown_list.append(unk_cut)
        label_list.append(l)
    result_df = pd.DataFrame({"label":label_list, "known":known_list, "unknown":unknown_list})
    return result_df    

def make_one_cut(df):
    label_list = []
    known_list = []
    unknown_list = []
    prob_index = []

    for irow in range(len(df)):
        l = df.iloc[irow, 0]
        kno = "\n".join( df.iloc[irow, 1] )
        unk = df.iloc[irow, 2]

        kno_cut = split_doc(kno, seq_word=100)
        unk_cut = split_doc(unk, seq_word=100)
        
        known_list.append(kno_cut[0])
        unknown_list.append(unk_cut[0])
        label_list.append(l)
        prob_index.append(irow)

    result_df = pd.DataFrame({"label":label_list, "known":known_list, "unknown":unknown_list, "prob_index":prob_index})
    return result_df

# %%
def make_model_and_tok(gpuid):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    _ = model.eval()
    _ = model.to(gpuid)

    for param in model.parameters():
        param.requires_grad = False

    pred_model = model.roberta
    enco_model = pred_model.embeddings.word_embeddings

    return (model, enco_model, pred_model, tokenizer)

def one_doc_embed(enc_model, pred_model, tokenizer, input_ids, input_mask, mask_n=1):
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
    return {"e": rec_embed, "p": rec_pred}

def make_KUEP(enco_model, pred_model, tokenizer, df, gpuid):
    result_list = []
    for i in tqdm(range(len(df))):
        encoded_text = tokenizer(df.iloc[i,1], truncation=True, padding="max_length", max_length=128)
        input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64).to(gpuid)
        input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64).to(gpuid)
        k_result = one_doc_embed( enco_model, pred_model, tokenizer, input_ids, input_mask, mask_n=1 )
        k_result["e"] = k_result["e"].cpu()
        k_result["p"] = k_result["p"].cpu()
        k_result["input_ids"] = input_ids.cpu()
        k_result["input_mask"] = input_mask.cpu()

        encoded_text = tokenizer(df.iloc[i,2], truncation=True, padding="max_length", max_length=128)
        input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.int64).to(gpuid)
        input_mask = torch.tensor(encoded_text["attention_mask"], dtype=torch.int64).to(gpuid)
        u_result = one_doc_embed( enco_model, pred_model, tokenizer, input_ids, input_mask, mask_n=1 )
        u_result["e"] = u_result["e"].cpu()
        u_result["p"] = u_result["p"].cpu()
        u_result["input_ids"] = input_ids.cpu()
        u_result["input_mask"] = input_mask.cpu()

        l = df.iloc[i,0]=="Y"
        result_list.append({"l": l, "k": k_result, "u": u_result})
    return result_list

# %%
def make_2013():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_13_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13",
                        train_split=["pan13_train"],
                        test_split=["pan13_test02"], known_as="list")

    # train_df = make_all_combo(pan_data13.get_train())
    # print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    # train_df.to_pickle(PATH_CLS / 'train_kucombo_only.pickle')

    train_df = make_doc_cut_list(pan_data13.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_cutlist.pickle')

    # test_df = make_one_cut(pan_data13.get_test())
    # print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    # test_df.to_pickle(PATH_CLS / 'test2_onecut.pickle')

    test_df = make_doc_cut_list(pan_data13.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test2_cutlist.pickle')

    return train_df, test_df

# %%
traindf, testdf = make_2013()

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_13_cls/train_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_13_cls/train_KUEP.pt" )

# %%
result = torch.load( "./data_pickle_cutcombo/pan_13_cls/train_KUEP.pt" )
result_combo = make_all_combo_KUEP(result)
torch.save( result_combo, "./data_pickle_cutcombo/pan_13_cls/train_KUEP_combo.pt" )

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_13_cls/test2_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_13_cls/test2_KUEP.pt" )

# %%
def make_2014e():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_14e_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-essays"],
                        test_split=["pan14_test01_english-essays"], known_as="list")

    # train_df = make_all_combo(pan_data14.get_train())
    # print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    # train_df.to_pickle(PATH_CLS / 'train_essays_kucombo_only.pickle')

    train_df = make_doc_cut_list(pan_data14.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_cutlist.pickle')

    # test1_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test1_df["label"] == "Y" ), np.sum( test1_df["label"] == "N" ) ) )
    # test1_df.to_pickle(PATH_CLS / 'test01_essays_onecut.pickle')

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-essays"],
                        test_split=["pan14_test02_english-essays"], known_as="list")

    # test2_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_essays_onecut.pickle')

    test2_df = make_doc_cut_list(pan_data14.get_test())
    print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    test2_df.to_pickle(PATH_CLS / 'test02_essays_cutlist.pickle')
    
    return train_df, test2_df

# %%
traindf, testdf = make_2014e()

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_14e_cls/train_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_14e_cls/train_KUEP.pt" )

# %%
result = torch.load( "./data_pickle_cutcombo/pan_14e_cls/train_KUEP.pt" )
result_combo = make_all_combo_KUEP(result)
torch.save( result_combo, "./data_pickle_cutcombo/pan_14e_cls/train_KUEP_combo.pt" )

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_14e_cls/test02_essays_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_14e_cls/test02_essays_KUEP.pt" )

# %%
def make_2014n():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_14n_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-novels"],
                        test_split=["pan14_test01_english-novels"], known_as="list")

    # train_df = make_all_combo(pan_data14.get_train())
    # print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    # train_df.to_pickle(PATH_CLS / 'train_novels_kucombo_only.pickle')

    train_df = make_doc_cut_list(pan_data14.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_cutlist.pickle')

    # test1_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test1_df["label"] == "Y" ), np.sum( test1_df["label"] == "N" ) ) )
    # test1_df.to_pickle(PATH_CLS / 'test01_novels_onecut.pickle')

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-novels"],
                        test_split=["pan14_test02_english-novels"], known_as="list")

    # test2_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_novels_onecut.pickle')

    test2_df = make_doc_cut_list(pan_data14.get_test())
    print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    test2_df.to_pickle(PATH_CLS / 'test02_novels_cutlist.pickle')

    return train_df, test2_df
# %%
train_df, test2_df = make_2014n()

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_14n_cls/train_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_14n_cls/train_KUEP.pt" )

# %%
result = torch.load( "./data_pickle_cutcombo/pan_14n_cls/train_KUEP.pt" )
result_combo = make_all_combo_KUEP(result)
torch.save( result_combo, "./data_pickle_cutcombo/pan_14n_cls/train_KUEP_combo170k.pt" )

# %%
def make_2015():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_15_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data15 = PANData(year="15",
                        train_split=["pan15_train"],
                        test_split=["pan15_test"], known_as="list")

    # train_df = make_all_combo(pan_data15.get_train())
    # print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    # train_df.to_pickle(PATH_CLS / 'train_kucombo_only.pickle')

    train_df = make_doc_cut_list(pan_data15.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_cutlist.pickle')

    test_df = make_one_cut(pan_data15.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test_onecut.pickle')

    test_df = make_doc_cut_list(pan_data15.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test_cutlist.pickle')

    return train_df, test_df
# %%
train_df, test_df = make_2015()

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_15_cls/train_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_15_cls/train_KUEP.pt" )

# %%
result = torch.load( "./data_pickle_cutcombo/pan_15_cls/train_KUEP.pt" )
result_combo = make_all_combo_KUEP(result)
torch.save( result_combo, "./data_pickle_cutcombo/pan_15_cls/train_KUEP_combo.pt" )

# %%
df = pd.read_pickle('./data_pickle_cutcombo/pan_15_cls/test_cutlist.pickle')
model, enco_model, pred_model, tokenizer = make_model_and_tok("cuda:0")
result = make_KUEP(enco_model, pred_model, tokenizer, df, "cuda:0")
torch.save( result, "./data_pickle_cutcombo/pan_15_cls/test_KUEP.pt" )

# %%

# %%
def make_all():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_all_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13",
                        train_split=["pan13_train"],
                        test_split=["pan13_test02"], known_as="list")

    train13_df = make_all_combo(pan_data13.get_train())
    

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-essays"],
                        test_split=["pan14_test01_english-essays"], known_as="list")

    train14e_df = make_all_combo(pan_data14.get_train())

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-novels"],
                        test_split=["pan14_test01_english-novels"], known_as="list")

    train14n_df = make_all_combo(pan_data14.get_train())

    pan_data15 = PANData(year="15",
                        train_split=["pan15_train"],
                        test_split=["pan15_test"], known_as="list")

    train15_df = make_all_combo(pan_data15.get_train())

    pan_all = pd.concat( [train13_df, train14e_df, train14n_df, train15_df] )

    print( (np.sum( pan_all["label"] == "Y" ), np.sum( pan_all["label"] == "N" ) ) )
    pan_all.to_pickle(PATH_CLS / 'train_kucombo_only.pickle')

# %%
make_all()
# %%
