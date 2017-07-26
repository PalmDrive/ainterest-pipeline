# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
from ailab.algo.algorithm import *
# from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def predict(x_data, w, b):
    # predict the labels using trained weights and biases
    # y = sign(xw + b)

    y_pred = np.matmul(x_data, w)
    for i_d in range(w.shape[1]):
        y_pred[:, i_d] = y_pred[:, i_d] + b[i_d]
    y_pred = np.sign(y_pred)

    y_pred[y_pred == -1] = 0

    return y_pred


def accuracy(y_data, y_pred):
    # calculate the accuracy between true labels and compared results

    cor_data = np.zeros(shape=y_data.shape[0], dtype=np.float64)

    for i_d in range(y_data.shape[0]):
        if not (False in (y_data[i_d] == y_pred[i_d])):
            cor_data[i_d] = 1

    return cor_data.mean()


def data_plot(y_train, y_test, y_train_pred, y_test_pred):
    l_train = list(range(y_train.shape[0]))
    l_test = list(range(y_test.shape[0]))
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


def train_label(data_all):
    # train model for each label

    # data
    x_train = data_all[0]
    y_train = data_all[1]
    label_id = data_all[2]
    label_num = data_all[3]

    # w_train, b_train = prim_tf(x_train, y_train, label_id, label_num)
    # w_train, b_train = prim_admm(x_train, y_train, label_id, label_num)
    # w_train, b_train = dual_dcd(x_train, y_train, label_id, label_num)
    w_train, b_train = dual_l1dcd(x_train, y_train, label_id, label_num)
    # w_train, b_train = dual_l1dcd_test(x_train, y_train, label_id, label_num)

    return [w_train, b_train]


def train_field(x_train, y_train, thread=4):
    # Get the shape of the data.
    num_features = x_train.shape[1]
    num_labels = y_train.shape[1]

    # Weights and biases with different columns for different labels
    w_field = np.zeros(shape=(num_features, num_labels), dtype=np.float64)
    b_field = np.zeros(shape=num_labels, dtype=np.float64)

    # prepare data for multi-thread training
    data_all_list = list()
    for i_d in range(num_labels):
        data_all_list.append([x_train, y_train[:, i_d], i_d, num_labels])

    # begin training for the whole field
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

    # reshape the data
    for i_d in range(num_labels):
        result = results[i_d]
        w_field[:, i_d] = result[0]
        b_field[i_d] = result[1]

    return w_field, b_field

