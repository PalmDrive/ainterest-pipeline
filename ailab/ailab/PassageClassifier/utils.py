import numpy as np

class DataGenerator(object):

    def __init__(self, batch_size = 50, shuffle=True, filepath):
        """
        initilization
        """
        self.num = 450000
        self.dim = 3000
        self.y_dim = 5
        self.datafile = filepath
        self.batch_size=batch_size
        self.shuffle=shuffle

    def __get_order(self, list_ids):
        """
        shuffle the indexes
        """
        index = np.arange(len(list_ids))
        if self.shuffle == True:
            np.random.shuffle(index)
        return index

    def __data_generation(self, labels, list_ids):
        """
        Generate data of batch_size samples
        """
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size, self.y_dim))

        for i, id in enumerate(list_ids):

            X[i,:] = np.load(self.datafile + id + ".npy")
            y[i] = labels[id]
        return X, y


    def generate(self, labels, list_ids):
        """
        generate batches of samples
        """
        while 1:
            index = self.__get_order(list_ids)
            # batch_step
            imax = int(len(index)/self.batch_size)
            for i in range(imax):
                list_ids_temp = [list_ids[k] for k in index[i*self.batch_size:(i+1)*self.batch_size]]

                X, y = self.__data_generation(labels, list_ids_temp)

                yield [X] *3, y
# useful function
def sparsify(y, num_class):
    """
    convert label array to binary label matrix(label starts at 1)
    """
    return np.array([[1 if y[i] == j+1 else 0 for j in range(num_class)]
                    for i in range(y.shape[0])])

import pandas as pd
import jieba

# load in preprocessed data (a Pandas Dataframe: with N rows of samples and 2 columns comprised
# of "content" and "label")
data = pd.read_msgpack('../content_and_label')

# load in stopwords and define passage segmentation functionstopwords = []
# stop_words.txt could be found
with open('../stop_words.txt', 'rb') as f:
    stopwords = f.read().decode('gbk').splitlines()

import re
# after segmentation, converting to pinyin and ignore speical symbols
def passageSeg(passage):
    '''
    Remove stopwords and \\n s, make segementation
    Args:
        passage: a string of single passage
    Return:
        a string of segmentation
    '''
    clean = []
    passage = re.sub(r"http\S+", "", passage)
    passage = passage.strip('\n').replace('\n', '').replace(' ','')
    words = jieba.cut(passage, cut_all=False)
    for word in words:
        if word not in stopwords:
             clean.append(word)
    return ' '.join(clean)


data = data.dropna(subset=['content'])
data['cleanedContent'] = data['content'].apply(passageSeg)

# convert text into pinyin as character level cnn take in one alphabet at a time as input data.
import pypinyin
from functools import reduce

def conv2pinyin(texts):
    pinyin_text = []
    abnormal = []
    for i in range(len(texts)):
        text = pypinyin.pinyin(texts[i], style=pypinyin.TONE3)
        try:
            text = reduce(lambda x,y: x+y, text)
            text = ''.join(text)
        except:
            abnormal.append(i)
            continue
        pinyin_text.append(text)
    return pinyin_text, abnormal

def re(fun, lst):
    if len(lst) == 1 :
        return lst[0]
    if lst == []:
        return False
    return fun(lst[0], re(fun, lst[1:]))

import numpy as np

all_letters = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

n_letters = len(all_letters)


def letterToIndex(letter):
    """
    'c' -> 2
    """
    return all_letters.find(letter)


def sets2tensors(clean_train, n_letters=n_letters, MAX_SEQUENCE_LENGTH=1000):
    """
    From lists of cleaned passages to np.array with shape(len(train),
        max_sequence_length, len(dict))
    Arg:
        obviously
    """
    m = len(clean_train)
    x_data = np.zeros((m, MAX_SEQUENCE_LENGTH, n_letters))
    for ix in range(m):
        for no, letter in enumerate(clean_train[ix]):
            if no >= 1000:
                break
            letter_index = letterToIndex(letter)
            if letter != -1:
                x_data[ix][no][letter_index]  = 1
            else:
                continue
    return x_data

def to_cat(labels, num_class, start):
    """
    Convert a categorical label to a vector : 4 -> [0, 0,0,1]
    """
    labels = [[1 if i == l else 0 for i in range(start, num_class+start)] for l in labels]
    return np.array(labels)
