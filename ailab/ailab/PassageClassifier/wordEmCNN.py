from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPool1D, Embedding
from keras.models import Model
from keras import regularizers

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
embedding_dir = "../result/filv3.vec"

texts = data['cleanedContent'].tolist()

labels = data['zixun'].tolist()

print('Found {0} texts and {1} labels'.format(len(texts), len(labels)))

import numpy as np
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

afterPadding = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', afterPadding.shape)
print('Shape of label tensor:', labels.shape)

# shuffle the data and split it into train and test datasets
indices = np.arange(afterPadding.shape[0])
np.random.shuffle(indices)
afterPadding = afterPadding[indices]
labels = labels[indices]
nb_val = int(VALIDATION_SPLIT * data.shape[0])
x_train = afterPadding[:-nb_val]
x_val = afterPadding[-nb_val:]
y_train = labels[:-nb_val]
y_val = labels[-nb_val:]

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

# load in pre-trained word embedding
from itertools import islice
embeddings_index = {}
with open(embedding_dir) as f:
    next(f)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
print('Number of word vectors is %s' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

# link embedding to wordindex, create embedding matrix
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
# define own metrics: precision recall and f1
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

# Model 1: with different conv windows

from keras.models import Sequential
from keras.layers import Activation, GlobalMaxPool1D, Merge, concatenate
from keras.callbacks import EarlyStopping

submodels = []
for kw in (3, 4, 5):    # kernel sizes
    submodel = Sequential()
    submodel.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=MAX_SEQUENCE_LENGTH,
                           trainable=False))
    submodel.add(Conv1D(32,
                        kw,
                        padding='valid',
                        activation='relu',
                        strides=1))
    submodel.add(Conv1D(32,
                        kw,
                        padding='valid',
                        activation='relu',
                        strides=1))
    
    submodel.add(GlobalMaxPool1D())
    submodels.append(submodel)
big_model = Sequential()
big_model.add(Merge(submodels, mode="concat"))
big_model.add(Dense(64))

big_model.add(Activation('relu'))
big_model.add(Dropout(0.5))
big_model.add(Dense(labels.shape[1]))
big_model.add(Activation('softmax'))
print('Compiling model')
big_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

big_model.summary()

hist = big_model.fit([x_train, x_train, x_train],
                     y_train,
                     batch_size=50,
                     epochs=10,
                     validation_data=([x_val, x_val, x_val], y_val),
                     callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=2)]
                     )

## The one with only one type of conv kernal.
# building embedding_layer
# from keras.layers import Embedding
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)

# sequences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequences_input)
# x = Conv1D(32, 3, activation='relu')(embedded_sequences)
# #x = MaxPool1D(5)(x)
# #x = Conv1D(64, 3, activation='relu')(x)
# #x = MaxPool1D(5)(x)
# x = Conv1D(16, 3, activation='relu')(x)
# #x = MaxPool1D(35)(x)
# x = Flatten()(x)
# x = Dropout(0.5)(x)
# #x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.5)(x)
# preds = Dense(labels.shape[1], activation='softmax')(x)

# model = Model(sequences_input, preds)
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
# #early_stopping = Ear

# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           epochs=10, batch_size=50)

# big_model.summary()