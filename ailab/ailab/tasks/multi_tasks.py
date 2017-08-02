# -*- coding:utf-8 -*-
import numpy as np
import ailab.data_process as utils
import glob
import pymysql


# If save data
from ailab import train

IF_SAVE = False

# If show results
IF_SHOW_RESULT = True

# If plot
IF_PLOT = False


class MultiClass:

    def __init__(self):

        # algorithm
        self.__algo = 'l1dcd'

        # If load dictionary
        self.__if_load_dict = False

        # If load model
        self.__if_load_model = False

        # dictionaries
        self.__dict_article = dict()
        self.__dict_label = dict()

        # trained model
        self.__model_list = []

        # If show classified results
        self.__if_show = IF_SHOW_RESULT

    def multi_classify(self, articlesstr_in, request, model_dir):
        # request = 'field', 'contentType', 'position'

        is_str = isinstance(articlesstr_in, str)

        if is_str:
            articlesstr = [articlesstr_in]
        else:
            articlesstr = articlesstr_in[:]

        # load dictionaries
        if not self.__if_load_dict:
            dict_article, dict_label = utils.load_dictionary(request)

            if len(dict_article) == 0 or len(dict_label) == 0:
                return []

            self.__dict_article = dict_article
            self.__dict_label = dict_label

            del dict_article
            del dict_label

            self.__if_load_dict = True

        # load model
        if not self.__if_load_model:

            # if model exist
            filename_all_chaotic = glob.glob(model_dir + request + "/*." + self.__algo)
            if len(filename_all_chaotic) == 0:
                print('Model for algorithm {0} does not exist yet. Train it if needed.'.format(self.__algo))
                return []

            # load trained model
            if self.__algo == 'libsvm':
                from ailab.libsvm.svmutil import svm_load_model

                model_num = len(filename_all_chaotic)
                model_list = []
                for i_d in range(model_num):
                    model_i = svm_load_model(model_dir + request + "/model_{0}.".format(i_d) + self.__algo)
                    model_list.append(model_i)
            else:
                w = np.loadtxt(model_dir + request + "/w_train." + self.__algo)
                b = np.loadtxt(model_dir + request + "/b_train." + self.__algo)
                model_list = [w, b]

            self.__model_list = model_list

            # delete
            del model_list

            self.__if_load_model = True

        # extract keywords
        articlesjieba = utils.articles_jieba(articlesstr)

        # convert string data to matrix
        x_data = utils.string_to_matrix_article(self.__dict_article, articlesjieba)

        # classification
        y_pred = train.predict(x_data, self.__model_list, self.__algo)

        # get labels
        labelsstr = utils.matrix_to_string_label(self.__dict_label, y_pred)

        # print classification results
        if self.__if_show:
            for i_d in range(len(labelsstr)):
                print(i_d, ':', labelsstr[i_d])

        if is_str:
            labelsstr_out = labelsstr[0]
        else:
            labelsstr_out = labelsstr[:]

        return labelsstr_out

    def algorithm(self, algo):
        # available algorithm
        algo_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

        # if invalid
        if not (algo in algo_available):
            print('Unsupported algorithm.')
            print('Training process halted.')
            return []

        # reset model
        self.__algo = algo
        self.__if_load_model = False
        self.__model_list = []


class MultiTrain:

    def __init__(self):

        # connect config
        self.__config = {'host': '127.0.0.1', 'user': 'myaccount', 'passwd': 'mypassword',
                         'db': 'database', 'charset': 'utf8'}

        # algorithm
        self.__algo = 'l1dcd'

        # parameters
        self.param = {'libsvm': '-t 2 -c 5', 'C': 20, 'D': 5, 'max_epoch': 10000, 'err': 1e-3}

        # options
        self.options = {'thread': 2}

        # If has trained
        self.__if_has_trained = False

        # If has load data
        self.__if_has_load_data = False

        # data
        self.__x_train = []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []

        # trained model
        self.__model_list = []

        # model prediction
        self.__y_train_pred = []
        self.__y_test_pred = []

        # If save after training
        self.__if_save = IF_SAVE

        # If save after training
        self.__if_plot = IF_PLOT

    def multi_train(self, request, output_dir):

        # request = 'field', 'contentType', 'position'

        # Get data
        if not self.__if_has_load_data:
            x_train, y_train, x_test, y_test, err_get = utils.get_data(request, self.__config)
            # x_train, y_train, x_test, y_test, err_get = get_data_test(200, 20, 60620, 36)  # very simple test data
            # x_train, y_train, x_test, y_test = get_data_file()

            # if error
            if err_get:
                print('Encountered an error when requesting data from the mySQL server.')
                print('Training ended.')
                return

            # save data to self
            self.__x_train = x_train
            self.__y_train = y_train
            self.__x_test = x_test
            self.__y_test = y_test

            # delete
            del x_train
            del y_train
            del x_test
            del y_test

            self.__if_has_load_data = True

        # train
        model_list = train.train_field(self.__x_train, self.__y_train, algo=self.__algo,
                                       param=self.param, thread=self.options['thread'])

        if len(model_list) == 0:
            # if training is invalid
            print('Training process returned invalid results.')
            print('Training ended.')
            return

        # here training is already OK
        self.__if_has_trained = True

        # save model to self
        self.__model_list = model_list

        # delete
        del model_list

        # model evaluation on train set.
        print('evaluating model on train set...')
        y_train_pred = train.predict(self.__x_train, self.__model_list, self.__algo)
        print("Accuracy on train:", train.accuracy(self.__y_train, y_train_pred))

        # model evaluation on test set.
        print('evaluating model on test set...')
        y_test_pred = train.predict(self.__x_test, self.__model_list, self.__algo)
        print("Accuracy on test:", train.accuracy(self.__y_test, y_test_pred))

        # save prediction to self
        self.__y_train_pred = y_train_pred
        self.__y_test_pred = y_test_pred

        # delete
        del y_train_pred
        del y_test_pred

        # if save model
        if self.__if_save:
            self.multi_save(request, output_dir)

        # if plot training results
        if self.__if_plot:
            self.plot()

    def connect(self, config):
        for key in config:
            self.__config[key] = config[key]

        # try to make a connection
        try:  # if succeed
            conn = pymysql.connect(host=config['host'], user=config['user'],
                                   passwd=config['passwd'], db=config['db'])
            conn.close()
            print('Connection is OK.')

        except pymysql.Error as error:
            # if failed: print error message
            code, message = error.args
            print(code, message)
            print('Connection is Invalid.')

    def algorithm(self, algo):
        # available algorithm
        algo_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

        if not (algo in algo_available):
            print('Unsupported algorithm.')
            print('Training process halted.')
            return []

        self.__algo = algo
        self.__if_has_trained = False
        self.__model_list = []
        self.__y_train_pred = []
        self.__y_test_pred = []

    def thread(self, thread):
        self.options['thread'] = thread

    def multi_save(self, request, output_dir):

        # if has not trained
        if not self.__if_has_trained:
            print('Has not trained a model.')
            print('Nothing to do.')
            return

        train.save_model(self.__model_list, self.__algo, request, output_dir)

    def plot(self):

        # if has not trained
        if not self.__if_has_trained:
            print('Has not trained a model.')
            print('Nothing to do.')
            return

        train.data_plot(self.__y_train, self.__y_test, self.__y_train_pred, self.__y_test_pred)
