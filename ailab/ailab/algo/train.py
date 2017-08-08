# -*- coding:utf-8 -*-
import time
# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import ailab.algo.algorithm as ag
import ailab.algo.data_process as dp
import matplotlib.pyplot as plt


def accuracy(y_data, y_predict):
    # calculate the accuracy between true labels and compared results

    y_size = y_data.shape

    cor_data = np.zeros(shape=y_size[0], dtype=np.float64)

    if len(y_size) == 1:
        cor_data = y_data == y_predict
    else:
        for i_d in range(y_size[0]):
            if not (False in (y_data[i_d] == y_predict[i_d])):
                cor_data[i_d] = 1

    return cor_data.mean()


def accuracy_plot(y_train, y_test, y_train_predict, y_test_predict):
    # get label number
    y_size = y_train.shape

    if len(y_size) == 1:  # only one label
        accuracy_plot_fig(y_train, y_test, y_train_predict, y_test_predict)
    else:  # multi labels
        for i_d in range(y_train.shape[1]):
            accuracy_plot_fig(y_train[:, i_d], y_test[:, i_d],
                              y_train_predict[:, i_d], y_test_predict[:, i_d])

    # show
    plt.show()


def accuracy_plot_fig(y_train, y_test, y_train_predict, y_test_predict):

    # get index
    l_train = list(range(y_train.shape[0]))
    l_test = list(range(y_test.shape[0]))

    # figure
    fig = plt.figure(figsize=(20, 10))

    # first axis: train set
    ax1 = fig.add_subplot(2, 1, 1)

    # data
    line_train, = ax1.plot(l_train, y_train, 'b', linewidth=3,
                           linestyle="--", label='train data')

    # predict
    line_train_predict, = ax1.plot(l_train, y_train_predict, 'r',
                                   linewidth=1, label='train predict')

    # legend
    ax1.legend(handles=[line_train, line_train_predict], loc=7)

    # second axis: test set
    ax2 = fig.add_subplot(2, 1, 2)

    # data
    line_test, = ax2.plot(l_test, y_test, 'b', linewidth=3,
                          linestyle="--", label='test data')

    # predict
    line_test_predict, = ax2.plot(l_test, y_test_predict, 'r',
                                  linewidth=1, label='test pred')

    # legend
    ax2.legend(handles=[line_test, line_test_predict], loc=7)


def save_model(model_list, algorithm, output_dir):

    if algorithm == 'libsvm':
        save_model_libsvm(model_list, output_dir)
        return

    # else: w and b
    np.savetxt(output_dir + '/w_train.' + algorithm, model_list[0])
    np.savetxt(output_dir + '/b_train.' + algorithm, model_list[1])


def save_model_libsvm(model_list, output_dir):
    # import library SVM
    from ailab.libsvm.svmutil import svm_save_model

    # for each model
    for i_d in range(len(model_list)):
        svm_save_model(output_dir + '/model_' + str(i_d) + '.' + 'libsvm', model_list[i_d])


def train_label(data_all):
    # train model for each label

    # data
    x_train = data_all[0]
    y_train = data_all[1]
    algorithm = data_all[2]
    param = data_all[3]
    label_id = data_all[4]
    label_num = data_all[5]

    # announcement
    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'
          .format(sum(y_train) / len(y_train), label_id + 1, label_num))

    model = []

    # timer begin
    time_st = time.time()

    # choose the algorithm, available: ['libsvm', 'l1dcd', 'dcd', 'admm']
    if algorithm == 'libsvm':
        model = ag.call_libsvm(x_train, y_train, param)
    if algorithm == 'l1dcd':
        model = ag.dual_l1dcd(x_train, y_train, param, label_id, label_num)
    if algorithm == 'dcd':
        model = ag.dual_dcd(x_train, y_train, param, label_id, label_num)
    if algorithm == 'admm':
        model = ag.prim_admm(x_train, y_train, param, label_id, label_num)

    # finish
    time_ed = time.time()

    # announcement
    print('Training process for label {0} in {1} has finished.'
          'Training algorithm: {2}. Training time: {3} s'
          .format(label_id + 1, label_num, algorithm, time_ed - time_st))

    return model


def train_field(x_train, y_train, algorithm, param, thread):  # unfinished
    # available algorithms
    algorithm_available = ['libsvm', 'l1dcd', 'dcd', 'admm']

    if not (algorithm in algorithm_available):
        print('Unsupported algorithm.')
        print('Training process halted.')
        return []

    # Get the shape of the data.
    num_features = x_train.shape[1]
    num_labels = y_train.shape[1]

    if algorithm == 'libsvm':
        # Convert to LIBSVM format

        # begin
        print('Converting data to LIBSVM format.')
        print('converting...')

        # doing
        x_train = dp.data_to_libsvm_x(x_train)
        y_train = dp.data_to_libsvm_y(y_train)

        # finish
        print('Successfully converted data to LIBSVM format.')

    # prepare data for multi-thread training
    data_all_list = list()

    if algorithm == 'libsvm':
        for i_d in range(num_labels):
            data_all_list.append([x_train, y_train[i_d], algorithm, param, i_d, num_labels])
    else:
        for i_d in range(num_labels):
            data_all_list.append([x_train, y_train[:, i_d], algorithm, param, i_d, num_labels])

    # begin training for the whole field
    print('Training process for all labels begin. Label number: {0}'.format(num_labels))

    # begin a timer
    time_st = time.time()

    # thread pool
    pool = ThreadPool(thread)

    # train each label
    results = pool.map(train_label, data_all_list)

    # close the pool
    pool.close()

    # end train
    time_ed = time.time()
    print('Training process for all labels has finished. Total training time: {0} s'
          .format(time_ed - time_st))

    # train results
    if algorithm == 'libsvm':  # models to form a list

        model_list = [model for model in results]

    else:  # Weights and biases with different columns for different labels

        # pre-allocate array
        w_field = np.zeros(shape=(num_features, num_labels), dtype=np.float64)
        b_field = np.zeros(shape=num_labels, dtype=np.float64)

        # for each label
        for i_d in range(len(results)):
            result = results[i_d]
            w_field[:, i_d] = result[0]
            b_field[i_d] = result[1]

        # to form a list
        model_list = [w_field, b_field]

    return model_list
