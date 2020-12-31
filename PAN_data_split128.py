# %%
from random import random
from pathlib import Path
import pandas as pd
import numpy as np
from PAN_data_converter import PANData

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
def make_combo(df):
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
        
        # all known combo
        for i in range(len(kno_cut)-1):
            for j in range(i+1, len(kno_cut)):
                if random() <= 0.5:
                    known_list.append(kno_cut[i])
                    unknown_list.append(kno_cut[j])
                    label_list.append("Y")
                    prob_index.append(irow)
                # else:
                #     unknown_list.append(kno_cut[i])
                #     known_list.append(kno_cut[j])
                #     label_list.append("Y")
        # all unknown combo
        for i in range(len(unk_cut)-1):
            for j in range(i+1, len(unk_cut)):
                if random() <= 0.5:
                    known_list.append(unk_cut[i])
                    unknown_list.append(unk_cut[j])
                    label_list.append("Y")
                    prob_index.append(irow)
                # else:
                #     unknown_list.append(unk_cut[i])
                #     known_list.append(unk_cut[j])
                #     label_list.append("Y")
        # all known-unknown combo
        for i in range(len(kno_cut)):
            for j in range(len(unk_cut)):
                if random() <= 0.5:
                    known_list.append(kno_cut[i])
                    unknown_list.append(unk_cut[j])
                    label_list.append(l)
                else:
                    unknown_list.append(kno_cut[i])
                    known_list.append(unk_cut[j])
                    label_list.append(l)
                prob_index.append(irow)
        # break
    result_df = pd.DataFrame({"label":label_list, "known":known_list, "unknown":unknown_list, "prob_index":prob_index})
    return result_df

# %%
pan_data14 = PANData(year="14",
                    train_split=["pan14_train_english-essays"],
                    test_split=["pan14_test01_english-essays"], known_as="list")

df = pan_data14.get_train()
# %%
train_df = make_combo(df)
# %%
np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" )
# %%
PATH_CLS = Path('./data_pickle_cutcombo/pan_14e_cls/')
PATH_CLS.mkdir(exist_ok=True)
train_df.to_pickle(PATH_CLS / 'train_essays.pickle')
# %%
test1_df = make_combo(pan_data14.get_test())
np.sum( test1_df["label"] == "Y" ), np.sum( test1_df["label"] == "N" )
# %%
PATH_CLS = Path('./data_pickle_cutcombo/pan_14e_cls/')
PATH_CLS.mkdir(exist_ok=True)
test1_df.to_pickle(PATH_CLS / 'test01_essays.pickle')
# %%
pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-essays"],
                        test_split=["pan14_test02_english-essays"], known_as="list")
test2_df = make_combo(pan_data14.get_test())
np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" )
# %%
PATH_CLS = Path('./data_pickle_cutcombo/pan_14e_cls/')
PATH_CLS.mkdir(exist_ok=True)
test2_df.to_pickle(PATH_CLS / 'test02_essays.pickle')
# %%
