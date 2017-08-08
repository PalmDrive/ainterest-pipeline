# -*- coding:utf-8 -*-
import glob
import numpy as np
import ailab.algo.data_process as utils
from ailab.algo import train
from ailab.algo.classify import predict
from ailab.db.db import DB


# If save data
IF_SAVE = False

# If show results
IF_SHOW_RESULT = True

# If plot
IF_PLOT = False


class MultiClass:

    def __init__(self):

        # algorithm
        self.__algorithm = 'l1dcd'

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

    def multi_classify(self, articles_str_in, model_dir):

        # string or list
        is_str = isinstance(articles_str_in, str)

        if is_str:
            articles_str = [articles_str_in]
        else:
            articles_str = articles_str_in[:]

        # load dictionaries
        if not self.__if_load_dict:
            dict_article, dict_label = utils.load_dictionary(model_dir)

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
            filename_all_chaotic = glob.glob(model_dir + "/*." + self.__algorithm)

            # if not
            if len(filename_all_chaotic) == 0:
                print('Model for algorithm {0} does not exist yet. Train it if needed.'.format(self.__algorithm))
                return []

            # load trained model
            if self.__algorithm == 'libsvm':
                # library SVM
                from ailab.libsvm.svmutil import svm_load_model

                # number
                model_num = len(filename_all_chaotic)

                # model as a list
                model_list = []

                # for each model
                for i_d in range(model_num):
                    model_i = svm_load_model(model_dir + "/model_{0}.".format(i_d) + self.__algorithm)
                    model_list.append(model_i)

            else:
                # weight and bias
                w = np.loadtxt(model_dir + "/w_train." + self.__algorithm)
                b = np.loadtxt(model_dir + "/b_train." + self.__algorithm)
                model_list = [w, b]

            self.__model_list = model_list

            # delete
            del model_list

            self.__if_load_model = True

        # extract keywords
        articles_jieba = utils.articles_to_jieba(articles_str)

        # convert string data to matrix
        x_data = utils.string_to_matrix_article(self.__dict_article, articles_jieba)

        # classification
        y_predict = predict(x_data, self.__model_list, self.__algorithm)

        # get labels
        labels_str = utils.matrix_to_string_label(self.__dict_label, y_predict)

        # print classification results
        if self.__if_show:
            for i_d in range(len(labels_str)):
                print(i_d, ':', labels_str[i_d])

        if is_str:
            labels_str_out = labels_str[0]
        else:
            labels_str_out = labels_str[:]

        return labels_str_out

    def algorithm(self, algorithm):
        # available algorithm
        algorithm_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

        # if invalid
        if not (algorithm in algorithm_available):
            print('Unsupported algorithm.')
            print('Training process halted.')
            return []

        # reset model
        self.__algorithm = algorithm
        self.__if_load_model = False
        self.__model_list = []


class MultiTrain:

    def __init__(self):

        # connect config
        self.__config = None

        # database for test connect
        self.__db = DB(config=None)

        # algorithm
        self.__algorithm = 'l1dcd'

        # parameters
        self.param = {'libsvm': '-t 2 -c 20 -g 0.001', 'C': 20,
                      'D': 5, 'max_epoch': 10000, 'err': 1e-3}

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
        self.__y_train_predict = []
        self.__y_test_predict = []

        # If save after training
        self.__if_save = IF_SAVE

        # If save after training
        self.__if_plot = IF_PLOT

    def multi_train(self, request, output_dir):

        # request = 'field', 'content_type', 'position'

        # test the connection
        connect_err = self.connect(self.__config)

        # if connection is invalid
        if connect_err:
            return

        # Get data
        if not self.__if_has_load_data:

            x_train, y_train, x_test, y_test, err_get = utils.get_data(request, self.__config, output_dir=output_dir)
            # x_train, y_train, x_test, y_test, err_get = get_data_test(200, 20, 60620, 36)  # very simple test data
            # x_train, y_train, x_test, y_test, err_get = utils.get_data_file()

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
        model_list = train.train_field(self.__x_train, self.__y_train, self.__algorithm,
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
        y_train_predict = predict(self.__x_train, self.__model_list, self.__algorithm)

        # model evaluation on test set.
        print('evaluating model on test set...')
        y_test_predict = predict(self.__x_test, self.__model_list, self.__algorithm)

        # print
        print("Accuracy on train:", train.accuracy(self.__y_train, y_train_predict))
        print("Accuracy on test:", train.accuracy(self.__y_test, y_test_predict))

        # save prediction to self
        self.__y_train_predict = y_train_predict
        self.__y_test_predict = y_test_predict

        # delete
        del y_train_predict
        del y_test_predict

        # if save model
        if self.__if_save:
            self.multi_save(output_dir)

        # if plot training results
        if self.__if_plot:
            self.plot()

    def connect(self, config=None):

        # set
        self.__config = self.__db.set_config(config)

        # test the connection via class DB
        connect_err = self.__db.test_connect()

        return connect_err

    def algorithm(self, algorithm):
        # available algorithm
        algorithm_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

        if not (algorithm in algorithm_available):
            print('Unsupported algorithm.')
            print('Training process halted.')
            return []

        # store
        self.__algorithm = algorithm

        # reset, data are not affected.
        self.__if_has_trained = False
        self.__model_list = []
        self.__y_train_predict = []
        self.__y_test_predict = []

    def thread(self, thread):
        self.options['thread'] = thread

    def multi_save(self, output_dir):

        # if has not trained
        if not self.__if_has_trained:
            print('Has not trained a model.')
            print('Nothing to do.')
            return

        train.save_model(self.__model_list, self.__algorithm, output_dir)

    def plot(self):

        # if has not trained
        if not self.__if_has_trained:
            print('Has not trained a model.')
            print('Nothing to do.')
            return

        train.accuracy_plot(self.__y_train, self.__y_test, self.__y_train_predict, self.__y_test_predict)
