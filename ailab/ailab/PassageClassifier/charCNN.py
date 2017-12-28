MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2

import pandas as pd
import jieba
content_and_label_path = '../content_and_label'
data = pd.read_msgpack(content_and_label_path)

# load in stopwords and define passage segmentation function
stopwords = []
stop_words_path = '../stop_words.txt'
with open(stop_words_path, 'rb') as f:
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

texts = data['cleanedContent'].tolist()

labels = data['zixun'].tolist()

print('Found {0} texts and {1} labels'.format(len(texts), len(labels)))

result = [(a,b) for a, b in zip(texts, labels) if a != '']
ttext, tlabels = map(list, zip(*result))

ttext, tlabels = [list(x) for x in zip(*result)]

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

test_texts, abn = conv2pinyin(ttext)

import numpy as np

labels = to_cat(tlabels, 2, 0)
#texts = conv2pinyin(texts)
texts = sets2tensors(test_texts)
print('Shape of data tensor:', texts.shape)
print('Shape of label tensor:', labels.shape)

# shuffle the data and split it into train and test datasets
indices = np.arange(texts.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]
nb_val = int(VALIDATION_SPLIT * texts.shape[0])
x_train = texts[:-nb_val]
x_val = texts[-nb_val:]
y_train = labels[:-nb_val]
y_val = labels[-nb_val:]

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

%reload_ext autoreload
%autoreload -l

## Model
num_classes = 2
n_letters = 70
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Activation, GlobalMaxPool1D, Conv1D, Dense, Dropout
from keras.callbacks import EarlyStopping, History
from keras.optimizers import SGD
from keras.layers import concatenate
submodels = []
inputs = []
for kw in (3, 4, 5):    # kernel sizes
    input1 = Input(shape=(1000, n_letters))
    inputs.append(input1)
    input1 = Conv1D(32, kw, padding='valid', activation='relu',
                    strides=1)(input1)

    input1 = GlobalMaxPool1D()(input1)
    submodels.append(input1)


model = concatenate(submodels)
#big_model.add(Merge(submodels, mode="concat"))
model = Dense(64, activation='relu')(model)
model = Dropout(0.5)(model)
output = Dense(num_classes, activation='softmax')(model)
model = Model(inputs, output)
print('Compiling model')
#opt = SGD(lr=1e-3)
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = History()
model.fit([x_train] * 3,
                     y_train,
                     batch_size=1,
                     epochs=5,
                     validation_data=([x_val] * 3, y_val),
                     callbacks=[history, EarlyStopping(monitor='val_loss', mode='min', patience=2)]
                     )

# Print train_acc and val_acc
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('f1')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
