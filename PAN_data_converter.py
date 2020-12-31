# %%
import os
import re
from pathlib import Path
import pandas as pd
# %%
class PANData(object):
    re1 = re.compile(r'  +')
    re2 = re.compile(r"(\\uf.{3})")

    def __init__(self, year, train_split, test_split, known_as="list"):
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
                if known_as == "list":
                    self.train_splits.append({"label": l, 'k_doc': k_docs, 'u_doc': u_doc})
                elif known_as == "str":
                    self.train_splits.append({"label": l, 'k_doc': " ".join(k_docs), 'u_doc': u_doc})
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
                if known_as == "list":
                    self.test_splits.append({"label": l, 'k_doc': k_docs, 'u_doc': u_doc})
                elif known_as == "str":
                    self.test_splits.append({"label": l, 'k_doc': " ".join(k_docs), 'u_doc': u_doc})
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
        # x = x.replace("\n", " xcrlfx ")
        # x = self.re2.sub(r" \1 ", x)
        x = x.encode("ascii", "ignore").decode("utf-8")
        # x = textacy.preprocess_text(x, fix_unicode=True, transliterate=True)
        # x = self.re1.sub(' ', x)
        return x

    def load_one_problem(self, problem_dir):
        doc_file_list = sorted(os.listdir(problem_dir))
        k_docs = []
        u_doc = None
        for doc_file in doc_file_list:
            with open(os.path.join(problem_dir, doc_file), encoding='utf-8-sig') as f:
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


def convert_pan13():
    PATH_CLS = Path('./data_pickle_trfm/pan_13_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data13 = PANData(year="13", train_split=["pan13_train"], test_split=["pan13_test01"], known_as="str")

    # pan_data13.get_train().to_csv(PATH_CLS / 'train.csv', header=True, index=False)
    # pan_data13.get_test().to_csv(PATH_CLS / 'test01.csv', header=True, index=False)
    pan_data13.get_train().to_pickle(PATH_CLS / 'train.pickle')
    pan_data13.get_test().to_pickle(PATH_CLS / 'test01.pickle')

    pan_data13 = PANData(year="13", train_split=["pan13_train"], test_split=["pan13_test02"], known_as="str")
    # pan_data13.get_test().to_csv(PATH_CLS / 'test02.csv', header=True, index=False)

    pan_data13.get_test().to_pickle(PATH_CLS / 'test02.pickle')

    print("ok")


def convert_pan14_essay():
    PATH_CLS = Path('data_pickle_trfm/pan_14e_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-essays"],
                         test_split=["pan14_test02_english-essays"], known_as="list")

    # pan_data14.get_train().to_csv(PATH_CLS / 'train_essays.csv', header=True, index=False)
    # pan_data14.get_test().to_csv(PATH_CLS / 'test02_essays.csv', header=True, index=False)
    pan_data14.get_train().to_pickle(PATH_CLS / 'train_essays.pickle')
    pan_data14.get_test().to_pickle(PATH_CLS / 'test02_essays.pickle')

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-essays"],
                         test_split=["pan14_test01_english-essays"], known_as="list")

    # pan_data14.get_test().to_csv(PATH_CLS / 'test01_essays.csv', header=True, index=False)
    pan_data14.get_test().to_pickle(PATH_CLS / 'test01_essays.pickle')

    print("ok")


def convert_pan14_novel():
    PATH_CLS = Path('data_pickle_trfm/pan_14n_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data14 = PANData(year="14",
                         train_split=["pan14_train_english-novels"],
                         test_split=["pan14_test02_english-novels"], known_as="str")

    # pan_data14.get_train().to_csv(PATH_CLS / 'train_novels.csv', header=True, index=False)
    # pan_data14.get_test().to_csv(PATH_CLS / 'test02_novels.csv', header=True, index=False)
    pan_data14.get_train().to_pickle(PATH_CLS / 'train_essays.pickle')
    pan_data14.get_test().to_pickle(PATH_CLS / 'test02_essays.pickle')

    print("ok")


def convert_pan15():
    PATH_CLS = Path('data_pickle_trfm/pan_15_cls/')
    PATH_CLS.mkdir(exist_ok=True)

    pan_data15 = PANData(year="15", train_split=["pan15_train"], test_split=["pan15_test"])

    # pan_data15.get_train().to_csv(PATH_CLS / "train.csv", index=False)
    # pan_data15.get_test().to_csv(PATH_CLS / "test.csv", index=False)
    pan_data15.get_train().to_pickle(PATH_CLS / 'train.pickle')
    pan_data15.get_test().to_pickle(PATH_CLS / 'test.pickle')

    print("ok")

# %%

if __name__ == "__main__":
    # convert_pan13()
    convert_pan14_essay()
    # convert_pan14_novel()
    # convert_pan15()

# %%
