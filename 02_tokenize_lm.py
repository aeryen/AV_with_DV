from fastai.text import *
import textacy



re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace("\n", " xcrlfx ")
    x = textacy.preprocess_text(x, fix_unicode=True, transliterate=True)
    x = re1.sub(' ', x)
    return x


def proc_texts(line):
    texts = ' xcrlfx xdocbeginx ' + line
    texts = fixup(texts)
    return texts


def load_line_by_line(filename):
    line_list = []
    with open(filename) as f:
        for line in f:
            line = proc_texts(line)
            line_list.append(line)

    tok = Tokenizer().process_all(line_list)

    return tok


if __name__ == "__main__":
    LM_PATH = Path('data_new/pan_all_lm/')
    LM_PATH.mkdir(exist_ok=True)
    chunksize = 24000

    tok_trn = load_line_by_line(LM_PATH / 'train.txt')
    tok_val = load_line_by_line(LM_PATH / 'test.txt')

    np.save(LM_PATH / 'train_tok.npy', tok_trn)
    np.save(LM_PATH / 'test_tok.npy', tok_val)

    # tok_trn = np.load(LM_PATH / 'tmp' / 'tok_trn.npy')
    # tok_val = np.load(LM_PATH / 'tmp' / 'tok_val.npy')

    freq = Counter(p for o in tok_trn for p in o)
    print(freq.most_common(25))

    max_vocab = 60000
    min_freq = 5

    itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '<pad>')
    itos.insert(0, '<unk>')

    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    print("size of vocab: " + str(len(itos)))

    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(LM_PATH / 'train_ids.npy', trn_lm)
    np.save(LM_PATH / 'test_ids.npy', val_lm)
    pickle.dump(itos, open(LM_PATH / 'itos.pkl', 'wb'))
