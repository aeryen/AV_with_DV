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
def make_2013():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_13_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13",
                        train_split=["pan13_train"],
                        test_split=["pan13_test02"], known_as="list")

    train_df = make_all_combo(pan_data13.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_kucombo_only.pickle')

    test_df = make_one_cut(pan_data13.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test2_onecut.pickle')

    test_df = make_doc_cut_list(pan_data13.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test2_cutlist.pickle')

# %%
make_2013()

# %%
def make_2014e():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_14e_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-essays"],
                        test_split=["pan14_test01_english-essays"], known_as="list")

    train_df = make_all_combo(pan_data14.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_essays_kucombo_only.pickle')

    # test1_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test1_df["label"] == "Y" ), np.sum( test1_df["label"] == "N" ) ) )
    # test1_df.to_pickle(PATH_CLS / 'test01_essays_onecut.pickle')

    # pan_data14 = PANData(year="14",
    #                     train_split=["pan14_train_english-essays"],
    #                     test_split=["pan14_test02_english-essays"], known_as="list")

    # test2_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_essays_onecut.pickle')

    # test2_df = make_doc_cut_list(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_essays_cutlist.pickle')

# %%
make_2014e()

# %%
def make_2014n():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_14n_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                        train_split=["pan14_train_english-novels"],
                        test_split=["pan14_test01_english-novels"], known_as="list")

    train_df = make_all_combo(pan_data14.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_novels_kucombo_only.pickle')

    # test1_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test1_df["label"] == "Y" ), np.sum( test1_df["label"] == "N" ) ) )
    # test1_df.to_pickle(PATH_CLS / 'test01_novels_onecut.pickle')

    # pan_data14 = PANData(year="14",
    #                     train_split=["pan14_train_english-novels"],
    #                     test_split=["pan14_test02_english-novels"], known_as="list")

    # test2_df = make_one_cut(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_novels_onecut.pickle')

    # test2_df = make_doc_cut_list(pan_data14.get_test())
    # print( ( np.sum( test2_df["label"] == "Y" ), np.sum( test2_df["label"] == "N" ) ) )
    # test2_df.to_pickle(PATH_CLS / 'test02_novels_cutlist.pickle')
# %%
make_2014n()

# %%
def make_2015():
    PATH_CLS = Path('./data_pickle_cutcombo/pan_15_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data15 = PANData(year="15",
                        train_split=["pan15_train"],
                        test_split=["pan15_test"], known_as="list")

    train_df = make_all_combo(pan_data15.get_train())
    print( (np.sum( train_df["label"] == "Y" ), np.sum( train_df["label"] == "N" ) ) )
    train_df.to_pickle(PATH_CLS / 'train_kucombo_only.pickle')

    test_df = make_one_cut(pan_data15.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test_onecut.pickle')

    test_df = make_doc_cut_list(pan_data15.get_test())
    print( ( np.sum( test_df["label"] == "Y" ), np.sum( test_df["label"] == "N" ) ) )
    test_df.to_pickle(PATH_CLS / 'test_cutlist.pickle')

# %%
make_2015()

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
