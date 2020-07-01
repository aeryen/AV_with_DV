from fastai.old.fastai.text import *

LM_DATA_PATH = Path('data/pan_all_lm/')
LM_MODEL_PATH = Path("model/pan_all_lm/")

CLAS_DATA_PATH = Path('data/pan_15_cls/')

itos = pickle.load((LM_DATA_PATH / 'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
vs = len(itos)

train_data = pd.read_pickle(str(CLAS_DATA_PATH / 'trn_ids.pkl'))
test_data = pd.read_pickle(str(CLAS_DATA_PATH / 'val_ids.pkl'))


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
em_sz, nh, nl = 400, 1150, 3


trn_dl = LanguageModelLoader(np.concatenate(ids_train), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(ids_test), bs, bptt)
md = LanguageModelData(LM_MODEL_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]

learner.load('lm1')

result = learner.predict()
learner.predict_array()

print("ok")
