# -*- coding:utf-8 -*-
import ailab.algo.algorithm as multialgo
import time
import numpy as np
# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import ailab.data_process as data_process


def predict(x_data, model, algo):
    # predict the labels using trained models

    if algo == 'libsvm':
        from ailab.libsvm.svmutil import svm_predict

        x_data_libsvm = data_process.data_to_libsvm_x(x_data)
        y_data_libsvm = data_process.data_to_libsvm_y(np.ones(shape=len(x_data_libsvm), dtype=np.float64))

        if isinstance(model, list):
            num_label = len(model)
            y_pred = np.zeros(shape=(len(x_data_libsvm), num_label), dtype=np.float64)
            for i_d in range(num_label):
                print('classifying data for label {0} in {1}...'.format(i_d + 1, num_label))
                y_pred_libsvm = svm_predict(y_data_libsvm, x_data_libsvm, model[i_d], '-q')
                y_pred[:, i_d] = y_pred_libsvm[0]
                print('Successfully classified data for label {0} in {1}. Algorithm: {2}.'
                      .format(i_d + 1, num_label, algo))
        else:
            print('classifying data...')
            y_pred = np.zeros(shape=len(x_data_libsvm), dtype=np.float64)
            y_pred_libsvm = svm_predict(y_data_libsvm, x_data_libsvm, model, '-q')
            y_pred[:] = y_pred_libsvm[0]
            print('Successfully classified data. Algorithm: {0}.'.format(algo))
    else:  # y = sign(xw + b)
        w = model[0]
        b = model[1]

        print('classifying data for all labels...')

        y_pred = np.matmul(x_data, w)

        w_size = w.shape

        if len(w_size) == 1:
            y_pred += b
        else:
            for i_d in range(w_size[1]):
                y_pred[:, i_d] = y_pred[:, i_d] + b[i_d]

        print('Successfully classified data for all labels. Algorithm: {0}.'.format(algo))

    y_pred = np.sign(y_pred)
    y_pred[y_pred == -1] = 0

    return y_pred


def accuracy(y_data, y_pred):
    # calculate the accuracy between true labels and compared results

    y_size = y_data.shape

    cor_data = np.zeros(shape=y_size[0], dtype=np.float64)

    if len(y_size) == 1:
        cor_data = y_data == y_pred
    else:
        for i_d in range(y_size[0]):
            if not (False in (y_data[i_d] == y_pred[i_d])):
                cor_data[i_d] = 1

    return cor_data.mean()

'''
def data_plot(y_train, y_test, y_train_pred, y_test_pred):
    l_train = list(range(y_train.shape[0]))
    l_test = list(range(y_test.shape[0]))

    y_size = y_train.shape

    if len(y_size) == 1:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        line_train, = ax1.plot(l_train, y_train, 'b', linewidth=3, linestyle="--", label='train data')
        line_train_pred, = ax1.plot(l_train, y_train_pred, 'r', linewidth=1, label='train pred')
        ax1.legend(handles=[line_train, line_train_pred], loc=7)
        ax2 = fig.add_subplot(2, 1, 2)
        line_test, = ax2.plot(l_test, y_test, 'b', linewidth=3, linestyle="--", label='test data')
        line_test_pred, = ax2.plot(l_test, y_test_pred, 'r', linewidth=1, label='test pred')
        ax2.legend(handles=[line_test, line_test_pred], loc=7)
    else:
        for i_d in range(y_train.shape[1]):
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(2, 1, 1)
            line_train, = ax1.plot(l_train, y_train[:, i_d], 'b', linewidth=3, linestyle="--", label='train data')
            line_train_pred, = ax1.plot(l_train, y_train_pred[:, i_d], 'r', linewidth=1, label='train pred')
            ax1.legend(handles=[line_train, line_train_pred], loc=7)
            ax2 = fig.add_subplot(2, 1, 2)
            line_test, = ax2.plot(l_test, y_test[:, i_d], 'b', linewidth=3, linestyle="--", label='test data')
            line_test_pred, = ax2.plot(l_test, y_test_pred[:, i_d], 'r', linewidth=1, label='test pred')
            ax2.legend(handles=[line_test, line_test_pred], loc=7)

    plt.show()
'''


def save_model(model_list, algo, request, output_dir):

    if algo == 'libsvm':
        from ailab.libsvm.svmutil import svm_save_model

        for i_d in range(len(model_list)):
            svm_save_model(output_dir + request + "/model_" + str(i_d) + "." + algo, model_list[i_d])
    else:
        np.savetxt(output_dir + request + "/w_train." + algo, model_list[0])
        np.savetxt(output_dir + request + "/b_train." + algo, model_list[1])


def train_label(data_all):
    # train model for each label

    # data
    x_train = data_all[0]
    y_train = data_all[1]
    algo = data_all[2]
    param = data_all[3]
    label_id = data_all[4]
    label_num = data_all[5]

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'
          .format(sum(y_train) / len(y_train), label_id + 1, label_num))

    model = []

    # algo_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

    timest = time.time()

    if algo == 'libsvm':
        model = multialgo.call_libsvm(x_train, y_train, param)
    if algo == 'l1dcd':
        model = multialgo.dual_l1dcd(x_train, y_train, param, label_id, label_num)
    if algo == 'dcd':
        model = multialgo.dual_dcd(x_train, y_train, param, label_id, label_num)
    if algo == 'admm':
        model = multialgo.prim_admm(x_train, y_train, param, label_id, label_num)

    timeed = time.time()

    print('Training process for label {0} in {1} has finished. Training algorithm: {2}. Training time: {3} s'
          .format(label_id + 1, label_num, algo, timeed - timest))

    return model


def train_field(x_train, y_train, algo, param, thread):  # unfinished
    # available algorithm
    algo_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

    if not (algo in algo_available):
        print('Unsupported algorithm.')
        print('Training process halted.')
        return []

    # Get the shape of the data.
    num_features = x_train.shape[1]
    num_labels = y_train.shape[1]

    if algo == 'libsvm':
        # Convert to LIBSVM format
        print('Converting data to LIBSVM format.')
        print('converting...')
        x_train = data_process.data_to_libsvm_x(x_train)
        y_train = data_process.data_to_libsvm_y(y_train)
        print('Successfully converted data to LIBSVM format.')

    # prepare data for multi-thread training
    data_all_list = list()
    if algo == 'libsvm':
        for i_d in range(num_labels):
            data_all_list.append([x_train, y_train[i_d], algo, param, i_d, num_labels])
    else:
        for i_d in range(num_labels):
            data_all_list.append([x_train, y_train[:, i_d], algo, param, i_d, num_labels])

    # begin training for the whole field
    print('Training process for all labels begin. Label number: {0}'.format(num_labels))
    timest = time.time()

    # thread pool
    pool = ThreadPool(thread)

    # train each label
    results = pool.map(train_label, data_all_list)

    # close the pool
    pool.close()

    # end train
    timeed = time.time()
    print('Training process for all labels has finished. Total training time: {0} s'.format(timeed - timest))

    # train results
    if algo == 'libsvm':
        model_list = [model for model in results]
    else:
        # Weights and biases with different columns for different labels
        w_field = np.zeros(shape=(num_features, num_labels), dtype=np.float64)
        b_field = np.zeros(shape=num_labels, dtype=np.float64)
        for i_d in range(len(results)):
            result = results[i_d]
            w_field[:, i_d] = result[0]
            b_field[i_d] = result[1]
        model_list = [w_field, b_field]

    return model_list
