import os
from fastai.text import *
import textacy


class PANData(object):
    re1 = re.compile(r'  +')

    def __init__(self, year, train_split, test_split, known_as_list=False):
        """
        this class go find the PAN dataset dir, check it, list it, load the whole txt.
        also read in truth.txt for labels.
        The results are in a Dataframe with 3 columns, k_doc, u_doc, label
        """
        p = os.path.abspath(os.path.dirname(__file__) + "/data/PAN" + str(year) + '/')
        self.year = year
        self.name = 'PAN' + str(year)

        dir_list = self.get_dir_list(p)

        self.train_splits = []

        for split_name in train_split:
            assert os.path.exists(os.path.join(p, split_name))

            train_labels = []
            with open(os.path.join(p, split_name, 'truth.txt')) as truth_file:
                for line in truth_file:
                    train_labels.append(line.strip().split())
            train_labels = dict(train_labels)

            for problem_dir in dir_list[split_name]:
                k_docs, u_doc = self.load_one_problem(problem_dir)
                l = train_labels[os.path.basename(problem_dir)]
                if known_as_list:
                    self.train_splits.append({"label": l, 'k_doc': k_docs, 'u_doc': u_doc})
                else:
                    for k in k_docs:
                        self.train_splits.append({"label": l, 'k_doc': k, 'u_doc': u_doc})

        self.test_splits = []

        for split_name in test_split:
            assert os.path.exists(os.path.join(p, split_name))

            test_labels = []
            with open(os.path.join(p, split_name, 'truth.txt')) as truth_file:
                for line in truth_file:
                    test_labels.append(line.strip().split())
            test_labels = dict(test_labels)

            for problem_dir in dir_list[split_name]:
                k_docs, u_doc = self.load_one_problem(problem_dir)
                l = test_labels[os.path.basename(problem_dir)]
                if known_as_list:
                    self.test_splits.append({"label": l, 'k_doc': k_docs, 'u_doc': u_doc})
                else:
                    for k in k_docs:
                        self.test_splits.append({"label": l, 'k_doc': k, 'u_doc': u_doc})

        col_names = ['label', 'k_doc', 'u_doc']
        self.train_splits = pd.DataFrame(self.train_splits, columns=col_names)
        self.test_splits = pd.DataFrame(self.test_splits, columns=col_names)

    def get_dir_list(self, dataset_dir):
        split_name = []
        split_dir_list = []

        for d in os.listdir(dataset_dir):
            problem_dir_list = []
            split_dir = os.path.join(dataset_dir, d)
            if os.path.isfile(split_dir):
                continue
            split_name.append(d)
            for problem in os.listdir(split_dir):
                if not (problem.startswith("EN") or problem.startswith("EE")):
                    continue
                problem_dir = os.path.join(split_dir, problem)
                if os.path.isfile(problem_dir):
                    continue
                problem_dir_list.append(problem_dir)
            split_dir_list.append(sorted(problem_dir_list))
        result = dict(zip(split_name, split_dir_list))

        return result

    def fixup(self, x):
        x = x.replace("\n", " xcrlfx ")
        # x = textacy.preprocess_text(x, fix_unicode=True, transliterate=True)
        # x = self.re1.sub(' ', x)
        return x

    def load_one_problem(self, problem_dir):
        doc_file_list = sorted(os.listdir(problem_dir))
        k_docs = []
        u_doc = None
        for doc_file in doc_file_list:
            with open(os.path.join(problem_dir, doc_file), encoding='utf-8') as f:
                if doc_file.startswith("known"):
                    doc = f.read()
                    doc = self.fixup(doc)
                    k_docs.append(doc)
                elif doc_file.startswith("unknown"):
                    u_doc = f.read()
                    u_doc = self.fixup(u_doc)
                else:
                    raise Exception(doc_file + " is not right!")
        return k_docs, u_doc

    def get_data(self):
        return self.train_splits, self.test_splits

    def get_train(self):
        return self.train_splits

    def get_test(self):
        return self.test_splits


def convert_all_lm():
    PATH_LM = Path('data_new/pan_all_lm/')
    PATH_LM.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13", train_split=["pan13_train"], test_split=["pan13_test01", "pan13_test02"])
    pan_data14 = PANData(year="14", train_split=["pan14_train_english-novels", "pan14_train_english-essays"],
                         test_split=["pan14_test01_english-essays", "pan14_test01_english-novels",
                                     "pan14_test02_english-essays", "pan14_test02_english-novels"])
    pan_data15 = PANData(year="15", train_split=["pan15_train"], test_split=["pan15_test"])

    all_txt = pan_data13.get_train()['u_doc']
    all_txt = all_txt.append(pan_data13.get_train()['k_doc'])

    all_txt = all_txt.append(pan_data13.get_test()['u_doc'])
    all_txt = all_txt.append(pan_data13.get_test()['k_doc'])

    all_txt = all_txt.append(pan_data14.get_train()['u_doc'])
    all_txt = all_txt.append(pan_data14.get_train()['k_doc'])

    all_txt = all_txt.append(pan_data14.get_test()['u_doc'])
    all_txt = all_txt.append(pan_data14.get_test()['k_doc'])

    all_txt = all_txt.append(pan_data15.get_train()['u_doc'])
    all_txt = all_txt.append(pan_data15.get_train()['k_doc'])

    all_txt = all_txt.append(pan_data15.get_test()['u_doc'])
    all_txt = all_txt.append(pan_data15.get_test()['k_doc'])

    uniq_doc = pd.unique(all_txt.values.ravel('K'))

    np.random.seed(42)
    rdn_idx = np.random.permutation(len(uniq_doc))

    uniq_doc = uniq_doc[rdn_idx]

    with open(PATH_LM / 'train.txt', 'w', encoding="utf-8") as train_file:
        for i in range(len(uniq_doc) - 80):
            train_file.write(uniq_doc[i] + "\n")

    with open(PATH_LM / 'test.txt', 'w', encoding="utf-8") as test_file:
        for i in range(len(uniq_doc) - 80, len(uniq_doc)):
            test_file.write(uniq_doc[i] + "\n")

    print("ok")


def convert_pan14_essay_lm():
    PATH_LM = Path('data_new/pan_14essay_lm/')
    PATH_LM.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14", train_split=["pan14_train_english-essays"],
                         test_split=["pan14_test01_english-essays", "pan14_test02_english-essays"]
                         # , known_as_list=True
                         )

    all_txt = pan_data14.get_train()['u_doc']
    all_txt = all_txt.append(pan_data14.get_train()['k_doc'])

    all_txt = all_txt.append(pan_data14.get_test()['u_doc'])
    all_txt = all_txt.append(pan_data14.get_test()['k_doc'])

    uniq_doc = pd.unique(all_txt.values.ravel('K'))

    np.random.seed(42)
    rdn_idx = np.random.permutation(len(uniq_doc))

    uniq_doc = uniq_doc[rdn_idx]

    with open(PATH_LM / 'train.txt', 'w', encoding="utf-8") as train_file:
        for i in range(len(uniq_doc) - 50):
            train_file.write(uniq_doc[i] + "\n")

    with open(PATH_LM / 'test.txt', 'w', encoding="utf-8") as test_file:
        for i in range(len(uniq_doc) - 50, len(uniq_doc)):
            test_file.write(uniq_doc[i] + "\n")

    print("ok")


def convert_pan15_lm():
    PATH_LM = Path('data_new/pan_15_lm/')
    PATH_LM.mkdir(exist_ok=True)

    pan_data15 = PANData(year="15", train_split=["pan15_train"], test_split=["pan15_test"])

    all_txt = pan_data15.get_train()['u_doc']
    all_txt = all_txt.append(pan_data15.get_train()['k_doc'])

    all_txt = all_txt.append(pan_data15.get_test()['u_doc'])
    all_txt = all_txt.append(pan_data15.get_test()['k_doc'])

    uniq_doc = pd.unique(all_txt.values.ravel('K'))

    np.random.seed(42)
    rdn_idx = np.random.permutation(len(uniq_doc))

    uniq_doc = uniq_doc[rdn_idx]

    with open(PATH_LM / 'train.txt', 'w', encoding="utf-8") as train_file:
        for i in range(len(uniq_doc) - 50):
            train_file.write(uniq_doc[i] + "\n")

    with open(PATH_LM / 'test.txt', 'w', encoding="utf-8") as test_file:
        for i in range(len(uniq_doc) - 50, len(uniq_doc)):
            test_file.write(uniq_doc[i] + "\n")

    print("ok")



def convert_pan13():
    PATH_CLS = Path('data/pan_13_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13", train_split=["pan13_train"], test_split=["pan13_test01"], known_as_list=True)

    # pan_data13.get_train().to_csv(PATH_CLS / 'train13.csv', header=True, index=False)
    # pan_data13.get_test().to_csv(PATH_CLS / 'test13_test01.csv', header=True, index=False)

    pan_data13.get_train().to_pickle(PATH_CLS / 'train13.pickle')
    pan_data13.get_test().to_pickle(PATH_CLS / 'test13_01.pickle')

    pan_data13 = PANData(year="13", train_split=["pan13_train"], test_split=["pan13_test02"], known_as_list=True)

    # pan_data13.get_test().to_csv(PATH_CLS / 'test13_test02.csv', header=True, index=False)
    pan_data13.get_test().to_pickle(PATH_CLS / 'test13_02.pickle')

    print("ok")


def convert_pan14_essay():
    PATH_CLS = Path('data/pan_14e_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-essays"],
                         test_split=["pan14_test02_english-essays"], known_as_list=True)

    pan_data14.get_train().to_pickle(PATH_CLS / 'rawdf_train.pickle')
    pan_data14.get_test().to_pickle(PATH_CLS / 'rawdf_test_02.pickle')

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-essays"],
                         test_split=["pan14_test01_english-essays"], known_as_list=True)

    pan_data14.get_test().to_pickle(PATH_CLS / 'rawdf_test_01.pickle')

    print("ok")


def convert_pan14_novel():
    PATH_CLS = Path('data/pan_14n_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-novels"],
                         test_split=["pan14_test02_english-novels"], known_as_list=True)

    pan_data14.get_train().to_pickle(PATH_CLS / 'train14.pickle')
    pan_data14.get_test().to_pickle(PATH_CLS / 'test14_02.pickle')

    print("ok")


def convert_pan15():
    PATH_CLS = Path('data_new/pan_15_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data15 = PANData(year="15", train_split=["pan15_train"], test_split=["pan15_test"])

    pan_data15.get_train().to_csv(PATH_CLS / "train.csv", index=False)
    pan_data15.get_test().to_csv(PATH_CLS / "test.csv", index=False)

    # pan_data15.get_train().to_pickle(PATH_CLS / 'train15.pickle')
    # pan_data15.get_test().to_pickle(PATH_CLS / 'test15.pickle')

    # with open(PATH_CLS / 'train15.txt', 'w', encoding="utf-8") as train_file:
    #     for i in range(len(pan_data15.get_train())):
    #         train_file.write(pan_data15.get_train()["label"][1] + "\n")
    #
    # with open(PATH_CLS / 'test15.txt', 'w', encoding="utf-8") as test_file:
    #     for i in range(len(pan_data15.get_test())):
    #         test_file.write(pan_data15.get_test()[i] + "\n")

    print("ok")


if __name__ == "__main__":
    # convert_pan14_essay()
    # convert_pan14_novel()
    convert_pan15()

    # convert_pan14_essay_lm()
    # convert_pan15_lm()
    # convert_all_lm()

