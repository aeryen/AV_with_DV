from fastai.text import *

# ==========================
# Load Data
# ==========================
LM_DATA_PATH = Path('data/pan_all_lm/')

ids_train = np.load(LM_DATA_PATH / 'ids_train.npy')
ids_test = np.load(LM_DATA_PATH / 'ids_test.npy')

itos = pickle.load(open(LM_DATA_PATH / 'itos.pkl', 'rb'))
vs = len(itos)

data_lm = TextLMDataBunch.from_ids(train_ids=ids_train, test_ids=ids_test, vocab=itos)

moms = (0.8, 0.7)

learn = language_model_learner(data_lm, AWD_LSTM)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-2), moms=moms)



# ==========================
# Load Weights
# ==========================
em_sz, nh, nl = 400, 1150, 3

PRE_MODEL_PATH = Path('model/wiki_pretrain')

wgts = torch.load(PRE_MODEL_PATH / 'fwd_wt103.h5',
                  map_location=lambda storage, loc: storage)

enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)

itos2 = pickle.load((PRE_MODEL_PATH / 'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})

new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i, w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r >= 0 else row_m

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

# ==========================
# ==========================

wd = 1e-7
bptt = 70
bs = 52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

MODEL_PATH = Path("model")

# input: super long id array
# bs: batch size
# bptt:
trn_dl = LanguageModelLoader(np.concatenate(ids_train), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(ids_test), bs, bptt)
# model path, pad_index, n_tok, train_data, val_data
md = LanguageModelData(MODEL_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])
# optimization function, embedding size, size of hidden, number of layer, drops
learner = md.get_model(opt_fn, em_sz, nh, nl,
                       dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)

learner.model.load_state_dict(wgts)

lr = 1e-3
lrs = lr

learner.fit(lrs / 2, 1, wds=wd, use_clr=(32, 2), cycle_len=1)

learner.save('lm_last_ft')
