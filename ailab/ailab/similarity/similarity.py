# -*- coding:utf-8 -*-
import time
import os
import glob
import ailab.similarity.similarity_utils as sim_util


class Similarity:

    def __init__(self, model_dir=None, config=None):

        # initial: no model nor id_list
        self.__model = None
        self.__id_list = list()

        # model directory
        self.__model_dir = model_dir

        # database connecting configuration
        self.__config = config

    def connect(self, config=None):

        # if no user config
        if config is None:
            config = self.__config

        # test the connection and set the config
        self.__config, connect_err = sim_util.test_connect(config)

        return connect_err

    def train(self, articles_raw, id_list, epoch=2):

        # python list
        if not isinstance(id_list, list):
            id_list = [id_list]

        # python list
        if not isinstance(articles_raw, list):
            articles_raw = [articles_raw]

        # the new id list should merge with the old one
        id_list_new = sim_util.merge_id(id_list, self.__id_list)

        # old model. may also be None.
        old_model = self.__model

        # if able to include all new documents
        if old_model is not None:
            total_num = old_model.corpus_count
            if len(id_list_new) > total_num:
                print('WARNING! Reserved tags not enough.')

        # train model
        model = sim_util.train_model(articles_raw, id_list, model=old_model, id_list_old=self.__id_list, epoch=epoch)

        # save this model to self
        self.__model = model

        # total number
        total_num = model.corpus_count

        # remove extra ids
        if len(id_list_new) > total_num:
            id_list_new = id_list_new[:total_num]

        # announcement
        print('id list is updated.')

        self.__id_list = id_list_new

    def base(self, total_num=None):

        # request recent articles from database
        articles_raw, id_list = sim_util.recent_articles(self.__config, total_num=total_num)

        # train a model
        self.train(articles_raw, id_list, epoch=20)

    def clear(self):

        # clear model and the id list
        self.__model = None
        self.__id_list = list()

    def set_dir(self, model_dir=None):

        # set the model directory
        if model_dir is not None:
            self.__model_dir = model_dir

    def save(self, output_dir=None):

        # if model is not trained or loaded yet
        if self.__model is None:
            print('Model does not exist.')
            return

        # model directory
        if output_dir is None:
            output_dir = self.__model_dir

        # time string as file name of the model
        time_str = str(time.time())

        # model path
        model_path = os.path.join(output_dir, time_str + '.model')

        # id list path
        ids_path = os.path.join(output_dir, time_str + '.model.ids')

        # save model
        self.__model.save(model_path)

        # save id list
        with open(ids_path, 'w') as f_write:
            for id_str in self.__id_list:
                f_write.write(id_str)
                f_write.write('\n')

    def load(self, model_path):

        self.__model, self.__id_list = sim_util.load_model(model_path)

    def __find_all_model(self, model_dir=None):

        # model directory
        if model_dir is None:
            model_dir = self.__model_dir

        # path pattern
        path_pattern = os.path.join(model_dir, '*.model')

        # all models
        path_all = glob.glob(path_pattern)

        # all file names
        file_name_all = list()

        # get the time when saved for each model
        time_list = list()

        # begin
        for d in path_all:

            # split to get full file name
            tmp = os.path.split(d)[1]

            # file name, because the last for chars are '.model'
            tmp_str = tmp[:-6]

            # time
            time_list.append(eval(tmp_str))

            # file names
            file_name_all.append(tmp)

        return file_name_all, time_list

    def load_newest(self, model_dir=None):

        # model directory
        if model_dir is None:
            model_dir = self.__model_dir

        # get all model files and the save time
        file_name_all, time_list = self.__find_all_model(model_dir)

        # biggest time
        file_name = file_name_all[time_list.index(max(time_list))]

        # load
        self.load(os.path.join(model_dir, file_name))

    def clear_old_model(self, model_dir=None):

        # model directory
        if model_dir is None:
            model_dir = self.__model_dir

        # get all model files and the save time
        file_name_all_to_remove, time_list = self.__find_all_model(model_dir)

        # leave out the newest model
        del file_name_all_to_remove[time_list.index(max(time_list))]

        # delete all other models as well as the id lists
        for d in file_name_all_to_remove:
            file_path = os.path.join(model_dir, d)
            os.remove(file_path)
            os.remove(file_path + '.ids')
            if os.path.exists(file_path + '.syn1neg.npy'):
                os.remove(file_path + '.syn1neg.npy')
            if os.path.exists(file_path + '.wv.syn0.npy'):
                os.remove(file_path + '.wv.syn0.npy')

    def similar(self, articles_raw, id_list, thres=0.67):

        # first step, train these articles
        self.train(articles_raw, id_list)

        # then, get all ids, so we could know the tags
        id_list_all = self.__id_list

        # then, find most similar articles
        top_n = 20

        # for only one article
        if not isinstance(id_list, list):

            # obtain the tag
            tag = id_list_all.index(id_list)

            # similar articles
            sim_articles = self.__model.docvecs.most_similar(tag, topn=top_n)

            # most similar articles
            most_sim_articles = sim_util.most_similar_articles(sim_articles, id_list_all, thres)

            return most_sim_articles

        # for many articles
        sim_articles_list = list()

        for id_str in id_list:

            # obtain the tag
            tag = id_list_all.index(id_str)

            # similar articles
            sim_articles = self.__model.docvecs.most_similar(tag, topn=top_n)

            # most similar articles
            most_sim_articles = sim_util.most_similar_articles(sim_articles, id_list_all, thres)

            sim_articles_list.append(most_sim_articles)

        return sim_articles_list
