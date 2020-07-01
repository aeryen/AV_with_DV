from fastai.old.fastai.text import *

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag


def get_texts(df):
    df["k_doc"] = f'CRLF {BOS} ' + df["k_doc"].astype(str)
    df["k_doc"] = Tokenizer().proc_all_mp(partition_by_cores(df["k_doc"]))

    df["u_doc"] = f'CRLF {BOS} ' + df["u_doc"].astype(str)
    df["u_doc"] = Tokenizer().proc_all_mp(partition_by_cores(df["u_doc"]))

    return df


def get_texts_combine(df):
    for i in range(len(df["k_doc"])):
        df["k_doc"][i] = f' CRLF {BOS} '.join(df["k_doc"][i])
    df["k_doc"] = f'CRLF {BOS} ' + df["k_doc"].astype(str)
    df["k_doc"] = Tokenizer().proc_all_mp(partition_by_cores(df["k_doc"]))

    df["u_doc"] = f'CRLF {BOS} ' + df["u_doc"].astype(str)
    df["u_doc"] = Tokenizer().proc_all_mp(partition_by_cores(df["u_doc"]))

    return df


def convert_pan13():
    CLAS_PATH = Path('data/pan_13_cls/')
    LM_PATH = Path('data/pan_all_lm/')
    chunksize = 24000

    df_trn = pd.read_pickle(CLAS_PATH / 'train13.pickle')
    df_val1 = pd.read_pickle(CLAS_PATH / 'test13_test01.pickle')
    df_val2 = pd.read_pickle(CLAS_PATH / 'test13_test02.pickle')

    tok_trn = get_texts_combine(df_trn)
    tok_val1 = get_texts_combine(df_val1)
    tok_val2 = get_texts_combine(df_val2)

    tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val1["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["k_doc"]])
    tok_val1["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan13_old():
    CLAS_PATH = Path('data/pan_13_cls_old/')
    LM_PATH = Path('data/pan_all_lm_old/')

    df_trn = pd.read_pickle(CLAS_PATH / 'train13.pickle')
    df_val1 = pd.read_pickle(CLAS_PATH / 'test13_test01.pickle')
    df_val2 = pd.read_pickle(CLAS_PATH / 'test13_test02.pickle')

    tok_trn = get_texts_combine(df_trn)
    tok_val1 = get_texts_combine(df_val1)
    tok_val2 = get_texts_combine(df_val2)

    tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val1["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["k_doc"]])
    tok_val1["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan14e_old():
    CLAS_PATH = Path('data/pan_14e_cls _old/')
    LM_PATH = Path('data/pan_all_lm_old/')

    df_trn = pd.read_pickle(CLAS_PATH / 'rawdf_train.pickle')
    df_val1 = pd.read_pickle(CLAS_PATH / 'rawdf_test_01.pickle')
    df_val2 = pd.read_pickle(CLAS_PATH / 'rawdf_test_02.pickle')
    #
    tok_trn = get_texts_combine(df_trn)
    tok_val1 = get_texts_combine(df_val1)
    tok_val2 = get_texts_combine(df_val2)

    tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    # tok_trn = pd.read_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    # tok_val1 = pd.read_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    # tok_val2 = pd.read_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val1["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["k_doc"]])
    tok_val1["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan14e_14elm():
    CLAS_PATH = Path('data/pan_14e_lm/')
    LM_PATH = Path('data/pan_14e_lm/')

    df_trn = pd.read_pickle(CLAS_PATH / 'rawdf_train.pickle')
    df_val1 = pd.read_pickle(CLAS_PATH / 'rawdf_test_01.pickle')
    df_val2 = pd.read_pickle(CLAS_PATH / 'rawdf_test_02.pickle')
    #
    tok_trn = get_texts_combine(df_trn)
    tok_val1 = get_texts_combine(df_val1)
    tok_val2 = get_texts_combine(df_val2)
    #
    tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    tok_trn = pd.read_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1 = pd.read_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2 = pd.read_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val1["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["k_doc"]])
    tok_val1["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan14e():
    CLAS_PATH = Path('data/pan_14e_cls/')
    LM_PATH = Path('data/pan_all_lm/')

    # df_trn = pd.read_pickle(CLAS_PATH / 'rawdf_train.pickle')
    df_val1 = pd.read_pickle(CLAS_PATH / 'rawdf_test_01.pickle')
    # df_val2 = pd.read_pickle(CLAS_PATH / 'rawdf_test_02.pickle')
    #
    # tok_trn = get_texts_combine(df_trn)
    tok_val1 = get_texts_combine(df_val1)
    # tok_val2 = get_texts_combine(df_val2)
    #
    # tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    # tok_val2.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    tok_trn = pd.read_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val1 = pd.read_pickle(str(CLAS_PATH / 'test01_tok.pkl'))
    tok_val2 = pd.read_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val1["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["k_doc"]])
    tok_val1["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val1["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val1.to_pickle(str(CLAS_PATH / 'test01_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan14n():
    CLAS_PATH = Path('data/pan_14n_cls/')
    LM_PATH = Path('data/pan_all_lm/')

    # df_trn = pd.read_pickle(CLAS_PATH / 'train14.pickle')
    # df_val1 = pd.read_pickle(CLAS_PATH / 'test14_02.pickle')
    #
    # tok_trn = get_texts_combine(df_trn)
    # tok_val1 = get_texts_combine(df_val1)
    #
    # tok_trn.to_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    # tok_val1.to_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    tok_trn = pd.read_pickle(str(CLAS_PATH / 'train_tok.pkl'))
    tok_val2 = pd.read_pickle(str(CLAS_PATH / 'test02_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val2["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["k_doc"]])
    tok_val2["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val2["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'train_ids.pkl'))
    tok_val2.to_pickle(str(CLAS_PATH / 'test02_ids.pkl'))


def convert_pan15():
    CLAS_PATH = Path('data/pan_15_cls/')
    LM_PATH = Path('data/pan_all_lm/')

    df_trn = pd.read_pickle(CLAS_PATH / 'train15.pickle')
    df_val = pd.read_pickle(CLAS_PATH / 'test15.pickle')

    tok_trn = get_texts(df_trn)
    tok_val = get_texts(df_val)

    tok_trn.to_pickle(str(CLAS_PATH / 'trn_tok.pkl'))
    tok_val.to_pickle(str(CLAS_PATH / 'val_tok.pkl'))

    itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    len(itos)

    tok_trn["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["k_doc"]])
    tok_trn["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_trn["u_doc"]])

    tok_val["k_doc"] = np.array([[stoi[o] for o in p] for p in tok_val["k_doc"]])
    tok_val["u_doc"] = np.array([[stoi[o] for o in p] for p in tok_val["u_doc"]])

    tok_trn.to_pickle(str(CLAS_PATH / 'trn_ids.pkl'))
    tok_val.to_pickle(str(CLAS_PATH / 'val_ids.pkl'))


if __name__ == "__main__":
    # convert_pan13()
    # convert_pan13_old()
    # convert_pan14e()
    # convert_pan14e_old()
    convert_pan14e_14elm()
    # convert_pan14n()
    # convert_pan15()
