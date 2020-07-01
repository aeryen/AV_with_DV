from fastai.old.fastai.text import *

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

CLAS_PATH = Path('data/pan_15_cls/')
LM_PATH = Path('data/PanLM/')

itos = pickle.load((LM_PATH / 'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

train_data = pd.read_pickle(str(CLAS_PATH / 'trn_ids.pkl'))
test_data = pd.read_pickle(str(CLAS_PATH / 'val_ids.pkl'))

bptt, em_sz, nh, nl = 70, 400, 1150, 3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48

train_data.loc[train_data["label"] == 'Y', "label"] = 1
train_data.loc[train_data["label"] == 'N', "label"] = 0

test_data.loc[test_data["label"] == 'Y', "label"] = 1
test_data.loc[test_data["label"] == 'N', "label"] = 0

c = int(train_data["label"].max()) + 1

trn_clas = [tuple(x) for x in train_data[["k_doc", "u_doc"]].values]
val_clas = [tuple(x) for x in test_data[["k_doc", "u_doc"]].values]

trn_labels = train_data["label"]
val_labels = test_data["label"]

trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: (max(len(trn_clas[x][0]), len(trn_clas[x][1]))), bs=bs // 2)
val_samp = SortSampler(val_clas, key=lambda x: (max(len(val_clas[x][0]), len(val_clas[x][1]))))
trn_dl = DataLoader(trn_ds, bs // 2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)

# part 1
# dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])

dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5

m = get_rnn_classifer(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                      layers=[em_sz * 3, 50, c], drops=[dps[4], 0.1],
                      dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip = 25.
learn.metrics = [accuracy]

lr = 3e-3
lrm = 2.6
lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])

lrs = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])

wd = 1e-7
wd = 0
learn.load_encoder('lm2_enc')

learn.freeze_to(-1)

learn.lr_find(lrs / 1000)
learn.sched.plot()

# wds weight decay parameters
learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))

learn.save('clas_0')
learn.load('clas_0')

learn.freeze_to(-2)
learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))

learn.save('clas_1')
learn.load('clas_1')

learn.unfreeze()
learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32, 10))

learn.sched.plot_loss()

learn.save('clas_2')
learn.sched.plot_loss()
