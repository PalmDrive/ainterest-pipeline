# -*- coding:utf-8 -*-
import numpy as np
import ailab.algo.data_process as dp


def predict(x_data, model, algorithm):
    # predict the labels using trained models

    if algorithm == 'libsvm':
        return predict_libsvm(x_data, model)

    # else: y = sign(xw + b)

    # model
    w = model[0]
    b = model[1]

    # announcement
    print('classifying data for all labels...')

    # without the bias
    y_predict = np.matmul(x_data, w)

    # get the spape
    w_size = w.shape

    # consider the bias
    if len(w_size) == 1:  # for one label
        y_predict += b
    else:  # for multi labels
        for i_d in range(w_size[1]):
            y_predict[:, i_d] = y_predict[:, i_d] + b[i_d]

    # announcement
    print('Successfully classified data for all labels. Algorithm: {0}.'
          .format(algorithm))

    # sign and -1 value to zero
    y_predict = np.sign(y_predict)
    y_predict[y_predict == -1] = 0

    return y_predict


def predict_libsvm(x_data, model):
    # import library SVM
    from ailab.libsvm.svmutil import svm_predict

    # x data to LIBSVM format
    x_data_lib = dp.data_to_libsvm_x(x_data)

    # y data is just a dummy array
    y_data_lib = dp.data_to_libsvm_y(np.ones(shape=len(x_data_lib), dtype=np.float64))

    # if model is a list, we need to calculate predicted values for all models
    if isinstance(model, list):

        # the number of models, which is the same as the number of labels
        num_label = len(model)

        # pre-allocate
        y_predict = np.zeros(shape=(len(x_data_lib), num_label), dtype=np.float64)

        # for each label
        for i_d in range(num_label):
            # announcement
            print('classifying data for label {0} in {1}...'.format(i_d + 1, num_label))

            # call the library SVM
            y_predict_lib = svm_predict(y_data_lib, x_data_lib, model[i_d], '-q')

            # list to numpy array
            y_predict[:, i_d] = y_predict_lib[0]

            # announcement
            print('Successfully classified data for label {0} in {1}. Algorithm: LIBSVM.'
                  .format(i_d + 1, num_label))

    else:  # if for a single model

        # announcement
        print('classifying data...')

        # pre-allocate
        y_predict = np.zeros(shape=len(x_data_lib), dtype=np.float64)

        # call library SVM
        y_pred_lib = svm_predict(y_data_lib, x_data_lib, model, '-q')

        # list to numpy array
        y_predict[:] = y_pred_lib[0]

        # announcement
        print('Successfully classified data. Algorithm: LIBSVM.')

    # sign and -1 value to zero
    y_predict = np.sign(y_predict)
    y_predict[y_predict == -1] = 0

    return y_predict
