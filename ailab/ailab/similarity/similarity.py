# -*- coding:utf-8 -*-
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import os
import yaml
from ailab.db.db import DB
from ailab.similarity.


class Similarity:

    def __init__(self):

        self.__id_list = list()
        self.__model = None
        self.__trained_num = 0
        self.__remain_num = 0

        self.__config = None

        # database for test connect
        self.__db = DB(config=None)

    def connect(self, config=None):

        # set
        self.__config = self.__db.set_config(config)

        # test the connection via class DB
        connect_err = self.__db.test_connect()

        return connect_err

    def get_recent_articles(self, total_article_num=40000):

        # get raw articles and ids
        id_list, articles_raw, err = self.__db.recent_articles(number=total_article_num)

    def train_model(self, docs):

        # if model has not been trained or loaded
        if self.__model is None:

            model = Doc2Vec(size=300, window=10, min_count=5, workers=11,
                            alpha=0.025, min_alpha=0.025)

            print('Model initialized.')

            time_st = time.time()

            model.build_vocab(docs)

            time_ed = time.time()

            print('Vocabulary built. Time: ' + str(time_ed - time_st) + ' s')

        else:

            model = self.__model

        print('Total document number: ' + str(len(docs)))

        epoch = 20

        print('Training this model by training data...')

        time_st = time.time()

        model.train(docs, total_examples=len(docs), epochs=epoch)

        time_ed = time.time()

        print('Training finished. Epochs: ' + str(epoch) + '. Total time: ' + str(time_ed - time_st) + ' s')

        self.__model = model

    def train_base_model(self, total_article_num=40000, config=None):

