# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
# import numba as nb
# from numba import jit


# Primal:
#                min  0.5||w||_2^2 + C sum(i: 1 - m) max(0, 1 - yi (w^T xi)) + D ||w||_1
#

# equivalent to: min  0.5||w||_2^2 + C/2 sum(i: 1 - m) [|1 - yi (w^T xi)| + (1 - yi (w^T xi))] + D||w||_1

# equivalent to: min  0.5||w||_2^2 + C/2 [||t||_1 - e^T t] + D||w||_1
#                s.t. t = Aw - e

#         where: A = (y1 x1, y2 x2, ..., ym xm)^T, e = (1, 1, ..., 1)^T
#    for bias b: xi -> (xi^T, 1)^T, w -> (w^T, b)^T


# Dual:
#                min  0.5||A^T alpha + beta||_2^2 - e^T alpha
#                s.t. 0 <= alpha_i <= C
#                     |beta_i| <= D

# BATCH_SIZE = 100  # The number of training examples to use per training step.
# In fact, we do not use minibatch.

# the parameter C of SVM
SVMC = 1

# the parameter D for L1 penalty
SVMD = 0.1

# Max epoch
MAX_EPOCH = 10000

# If show total loss
SHOW_LOSS = True


# @jit(nb.float64(nb.float64[:, :], nb.float64[:], nb.float64[:], nb.int64, nb.float64[:]),
#      nopython=True, cache=True)
def loss_label(x_data, y_data, w_train, b_train, zero_par):

    # raw
    y_raw = np.zeros(shape=x_data.shape[0], dtype=np.float64)
    for i_d in range(x_data.shape[0]):
        y_raw[i_d] = np.dot(x_data[i_d], w_train)
    y_raw = y_raw + b_train

    # SVM L1 loss function
    regularization_loss = 0.5 * np.sum(np.square(w_train))

    # Hinge loss
    hinge_loss = np.sum(np.maximum(zero_par, 1 - y_data * y_raw))

    # L1 penalty
    l1_loss = np.sum(np.abs(w_train))

    # total loss
    svm_loss = regularization_loss + SVMC * hinge_loss + SVMD * l1_loss

    return svm_loss


def shrink(y, nu):
    # shrinkage
    x = np.zeros(shape=y.shape, dtype=np.float64)
    p = np.abs(y) > nu
    x[p] = y[p] - nu * np.sign(y[p])
    return x


def dual_l1dcds(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: dual coordinate descent with Random Permutation
    # reference: A Dual Coordinate Descent Method for Large-scale Linear SVM [Hsieh et al, 2008, ICML]
    # fast and available for high dimentional data
    # we add a L1 penalty (SVMD not zero) in order to get a sparse weight vector
    # it could take about 3 hours for training the whole field 'labelledField', if we set the err_loss to be 1e-6
    # maybe we could consider the shrinking to speed up. reference: [Hsieh et al, 2008, ICML]

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # train data
    x_data = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1), dtype=np.float64)
    x_data[:, :-1] = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-8

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'.format(y_data.mean(), label_id + 1, label_num))

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # parameters
    q_par = np.linalg.norm(x_data, ord=2, axis=1)**2
    m_par = SVMD * np.ones(shape=size_dim, dtype=np.float64)
    u_par = SVMC
    zero_par = np.zeros(shape=size_data, dtype=np.float64)

    # initialize variables
    w_train = np.zeros(shape=size_dim, dtype=np.float64)
    a_train = np.zeros(shape=size_data, dtype=np.float64)

    # initial loss
    svm_loss_old = loss_label(x_data, y_data, w_train, 0, zero_par)

    # one key parameter
    p_star = 0

    old_list = list(range(size_data))

    Mbar = np.Inf
    mbar = -np.Inf
    err_shrink = 1e-4

    # train begin
    count_epoch = 0
    timest = time.time()

    # train
    while count_epoch < MAX_EPOCH:

        M = 0
        m = 0

        new_list = old_list.copy()
        new_list_shuffle = new_list.copy()
        np.random.shuffle(new_list_shuffle)

        for i_d in new_list_shuffle:
            ai_old = a_train[i_d]
            g = y_data[i_d] * np.dot(x_data[i_d], w_train) - 1
            pg = 0
            if ai_old == 0:
                if g > Mbar:
                    new_list.remove(i_d)
                elif g < 0:
                    pg = g
            elif ai_old == u_par:
                if g < mbar:
                    new_list.remove(i_d)
                elif g > 0:
                    pg = g
            else:
                pg = g

            M = np.maximum(M, g)
            m = np.minimum(m, g)

            if not np.abs(pg) == 0:
                ai = np.minimum(np.maximum(ai_old - g / q_par[i_d], 0), u_par)
                a_train[i_d] = ai
                p_star -= (ai - ai_old) * y_data[i_d] * x_data[i_d]
                p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
                w_train = p_train - p_star

        old_list = new_list.copy()

        if M - m < err_shrink:
            if len(old_list) != size_data:
                old_list = list(range(size_data))
                Mbar = np.Inf
                mbar = -np.Inf
                continue

        if M > 0:
            Mbar = M
        else:
            Mbar = np.Inf

        if m < 0:
            mbar = m
        else:
            mbar = -np.Inf

        count_epoch += 1

        # calculate loss
        if count_epoch % 10 == 0:
            p_star = 0
            for i_d in range(size_data):
                p_star -= a_train[i_d] * y_data[i_d] * x_data[i_d]
            p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
            w_train = p_train - p_star
            svm_loss = loss_label(x_data, y_data, w_train, 0, zero_par)
            if SHOW_LOSS:
                print('Training loss for label {0} in {1}: {2}th epoch of {3} - {4}'
                      .format(label_id + 1, label_num, count_epoch, MAX_EPOCH, svm_loss))

            # if meet the accuracy condition
            if np.abs(svm_loss - svm_loss_old) / np.minimum(svm_loss, svm_loss_old) < err_loss:
                break

            svm_loss_old = svm_loss
    # end train

    timeed = time.time()
    print('Training process for label {0} in {1} has finished. Training time: {2} s'
          .format(label_id + 1, label_num, timeed - timest))

    b_train = w_train[-1]
    w_train = w_train[:-1]

    return w_train, b_train


def dual_l1dcd_test(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: dual coordinate descent with Random Permutation
    # reference: A Dual Coordinate Descent Method for Large-scale Linear SVM [Hsieh et al, 2008, ICML]
    # fast and available for high dimentional data
    # we add a L1 penalty (SVMD not zero) in order to get a sparse weight vector
    # it could take about 3 hours for training the whole field 'labelledField', if we set the err_loss to be 1e-6
    # maybe we could consider the shrinking to speed up. reference: [Hsieh et al, 2008, ICML]

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # train data
    x_data = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1), dtype=np.float64)
    x_data[:, :-1] = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-3

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'.format(y_data.mean(), label_id + 1, label_num))

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # parameters
    # a_par = np.matmul(np.diag(y_data), x_data)
    q_par = np.linalg.norm(x_data, ord=2, axis=1)**2
    m_par = SVMD * np.ones(shape=size_dim, dtype=np.float64)
    u_par = SVMC
    zero_par = np.zeros(shape=size_data, dtype=np.float64)
    # one_par = np.ones(shape=size_data, dtype=np.float64)

    # initialize variables
    w_train = np.zeros(shape=size_dim, dtype=np.float64)
    a_train = np.zeros(shape=size_data, dtype=np.float64)

    # initial loss
    svm_loss_old = loss_label(x_data, y_data, w_train, 0, zero_par)

    # key parameters
    p_star = 0
    grad_to_q = np.Inf * np.ones(shape=size_data, dtype=np.float64)

    # coordinate number
    max_coord = 1000
    sub_iterator = 10

    # train begin
    count_epoch = 0
    timest = time.time()

    # train
    while count_epoch < MAX_EPOCH:

        for i_d in np.random.permutation(size_data):
            ai_old = a_train[i_d]
            grad_to_q[i_d] = (y_data[i_d] * np.dot(x_data[i_d], w_train) - 1) / q_par[i_d]
            if ai_old == 0:
                pg = np.minimum(grad_to_q[i_d], 0)
            elif ai_old == u_par:
                pg = np.maximum(grad_to_q[i_d], 0)
            else:
                pg = grad_to_q[i_d]
            if not np.abs(pg) == 0:
                ai = np.minimum(np.maximum(ai_old - grad_to_q[i_d], 0), u_par)
                a_train[i_d] = ai
                p_star -= (ai - ai_old) * y_data[i_d] * x_data[i_d]
                p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
                w_train = p_train - p_star

        for no_use in range(sub_iterator):
            g_sorted_index = sorted(range(size_data), key=lambda k: np.abs(grad_to_q[k]), reverse=True)
            count_coord = 0
            for i_d in g_sorted_index:
                if count_coord > max_coord:
                    break
                ai_old = a_train[i_d]
                grad_to_q[i_d] = (y_data[i_d] * np.dot(x_data[i_d], w_train) - 1) / q_par[i_d]
                if ai_old == 0:
                    pg = np.minimum(grad_to_q[i_d], 0)
                elif ai_old == u_par:
                    pg = np.maximum(grad_to_q[i_d], 0)
                else:
                    pg = grad_to_q[i_d]
                if not np.abs(pg) == 0:
                    ai = np.minimum(np.maximum(ai_old - grad_to_q[i_d], 0), u_par)
                    a_train[i_d] = ai
                    p_star -= (ai - ai_old) * y_data[i_d] * x_data[i_d]
                    p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
                    w_train = p_train - p_star
                    count_coord += 1

        count_epoch += 1

        # calculate loss
        if count_epoch % 50 == 0:
            p_star = 0
            for i_d in range(size_data):
                p_star -= a_train[i_d] * y_data[i_d] * x_data[i_d]
            p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
            w_train = p_train - p_star
            svm_loss = loss_label(x_data, y_data, w_train, 0, zero_par)
            if SHOW_LOSS:
                print('Training loss for label {0} in {1}: {2}th epoch of {3} - {4}'
                      .format(label_id + 1, label_num, count_epoch, MAX_EPOCH, svm_loss))

            # if meet the accuracy condition
            if np.abs(svm_loss - svm_loss_old) / np.minimum(svm_loss, svm_loss_old) < err_loss:
                break

            svm_loss_old = svm_loss
    # end train

    timeed = time.time()
    print('Training process for label {0} in {1} has finished. Training time: {2} s'
          .format(label_id + 1, label_num, timeed - timest))

    b_train = w_train[-1]
    w_train = w_train[:-1]

    return w_train, b_train


# @jit(nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:, :], nb.float64[:], nb.int64, nb.int64),
#      nopython=True, cache=True)
def dual_l1dcd(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: dual coordinate descent with Random Permutation
    # reference: A Dual Coordinate Descent Method for Large-scale Linear SVM [Hsieh et al, 2008, ICML]
    # fast and available for high dimentional data
    # we add a L1 penalty (SVMD not zero) in order to get a sparse weight vector
    # it could take about 3 hours for training the whole field 'labelledField', if we set the err_loss to be 1e-6
    # maybe we could consider the shrinking to speed up. reference: [Hsieh et al, 2008, ICML]

    print('Training process for label', label_id + 1, 'in', label_num)

    # train data
    x_data = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1), dtype=np.float64)
    x_data[:, :-1] = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-3

    # Ratio of the TRUE data
    print('Ratio of True on train set:', y_data.mean(), 'for label', label_id + 1, 'in', label_num)

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # parameters
    q_par = np.zeros(shape=size_data, dtype=np.float64)
    for i_d in range(size_data):
        q_par[i_d] = np.dot(x_data[i_d], x_data[i_d])
    m_par = SVMD * np.ones(shape=size_dim, dtype=np.float64)
    u_par = np.float64(SVMC)
    zero_par = np.zeros(shape=size_data, dtype=np.float64)

    # initialize variables
    w_train = np.zeros(shape=size_dim, dtype=np.float64)
    a_train = np.zeros(shape=size_data, dtype=np.float64)

    # initial loss
    svm_loss_old = loss_label(x_data, y_data, w_train, 0, zero_par)

    # one key parameter
    p_star = np.zeros(shape=size_dim, dtype=np.float64)
    index_list = np.arange(size_data)

    # train begin
    count_epoch = 0
    # timest = time.time()

    # train
    while count_epoch < MAX_EPOCH:
        np.random.shuffle(index_list)
        for i_d in index_list:
            ai_old = a_train[i_d]
            g = y_data[i_d] * np.dot(x_data[i_d], w_train) - 1
            if ai_old == 0:
                pg = np.minimum(g, 0)
            elif ai_old == u_par:
                pg = np.maximum(g, 0)
            else:
                pg = g
            if np.abs(pg) != 0:
                ai = np.minimum(np.maximum(ai_old - g / q_par[i_d], 0), u_par)
                a_train[i_d] = ai
                p_star -= (ai - ai_old) * y_data[i_d] * x_data[i_d]
                p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
                w_train = p_train - p_star

        count_epoch += 1

        # calculate loss
        if count_epoch % 10 == 0:
            p_star[:] = 0
            for i_d in range(size_data):
                p_star -= a_train[i_d] * y_data[i_d] * x_data[i_d]
            p_train = np.minimum(np.maximum(p_star, -m_par), m_par)
            w_train = p_train - p_star
            svm_loss = loss_label(x_data, y_data, w_train, 0, zero_par)
            if SHOW_LOSS:
                print('Training loss for label', label_id + 1, 'in', label_num, ':', count_epoch, 'th epoch of',
                      MAX_EPOCH, '-', svm_loss)

            # if meet the accuracy condition
            if np.abs(svm_loss - svm_loss_old) / np.minimum(svm_loss, svm_loss_old) < err_loss:
                break

            svm_loss_old = svm_loss
    # end train

    # timeed = time.time()
    print('Training process for label', label_id + 1, 'in', label_num, 'has finished.')
    # print('Training process for label', label_id + 1, 'in', label_num, 'has finished. Training time:',
    #       timeed - timest, 's')

    b_train = w_train[-1]
    w_train = w_train[:-1]

    return w_train, b_train


def dual_dcd(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: dual coordinate descent with Random Permutation
    # reference: A Dual Coordinate Descent Method for Large-scale Linear SVM [Hsieh et al, 2008, ICML]
    # fast and available for high dimentional data

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # train data
    x_data = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1), dtype=np.float64)
    x_data[:, :-1] = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-8

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'.format(y_data.mean(), label_id + 1, label_num))

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # parameters
    q_par = np.linalg.norm(x_data, ord=2, axis=1)**2
    # d_par = np.zeros(shape=size_data, dtype=np.float64) + 1 / (2 * SVMC)
    d_par = np.zeros(shape=size_data, dtype=np.float64)
    q_par = q_par + d_par
    # u_par = np.inf
    u_par = SVMC
    zero_par = np.zeros(shape=size_data, dtype=np.float64)

    # initialize variables
    w_train = np.zeros(shape=size_dim, dtype=np.float64)
    a_train = np.zeros(shape=size_data, dtype=np.float64)

    # initial loss
    svm_loss_old = loss_label(x_data, y_data, w_train, 0, zero_par)

    # train begin
    count_epoch = 0
    timest = time.time()

    # train
    while count_epoch < MAX_EPOCH:
        for i_d in np.random.permutation(size_data):
            ai_old = a_train[i_d]
            g = y_data[i_d] * np.dot(w_train, x_data[i_d]) - 1 + d_par[i_d] * ai_old
            if ai_old == 0:
                pg = np.minimum(g, 0)
            elif ai_old == u_par:
                pg = np.maximum(g, 0)
            else:
                pg = g
            if not np.abs(pg) == 0:
                ai = np.minimum(np.maximum(ai_old - g / q_par[i_d], 0), u_par)
                a_train[i_d] = ai
                w_train += (ai - ai_old) * y_data[i_d] * x_data[i_d]

        count_epoch += 1

        # calculate loss
        if count_epoch % 10 == 0:
            w_train[:] = 0
            for i_d in range(size_data):
                w_train += a_train[i_d] * y_data[i_d] * x_data[i_d]
            svm_loss = loss_label(x_data, y_data, w_train, 0, zero_par)
            if SHOW_LOSS:
                print('Training loss for label {0} in {1}: {2}th epoch of {3} - {4}'
                      .format(label_id + 1, label_num, count_epoch, MAX_EPOCH, svm_loss))

            # if meet the accuracy condition
            if np.abs(svm_loss - svm_loss_old) / np.minimum(svm_loss, svm_loss_old) < err_loss:
                break

            svm_loss_old = svm_loss
    # end train

    timeed = time.time()
    print('Training process for label {0} in {1} has finished. Training time: {2} s'
          .format(label_id + 1, label_num, timeed - timest))

    b_train = w_train[-1]
    w_train = w_train[:-1]

    return w_train, b_train


def prim_admm(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: ADMM for prim problem
    # slow and could not handle data with features more than 40k

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # train data
    x_data = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-8

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'.format(y_data.mean(), label_id + 1, label_num))

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # parameters
    mu = 100
    nu = 100
    c_half = SVMC / 2
    a_par = np.matmul(np.diag(y_data), x_data)
    zero_par = np.zeros(shape=size_data, dtype=np.float64)
    one_par = np.ones(shape=size_data, dtype=np.float64)
    w_par1 = np.linalg.inv(np.eye(size_dim) * (1 / c_half + 1 / nu) + np.matmul(a_par.T, a_par) / mu)
    w_par2 = (a_par / mu).T
    b_par1 = 1 / np.dot(y_data, y_data)
    b_par2 = y_data

    # initialize variables
    w_train = np.zeros(shape=size_dim, dtype=np.float64)
    b_train = np.zeros(shape=(), dtype=np.float64)
    t_train = np.zeros(shape=size_data, dtype=np.float64)
    z_train = np.zeros(shape=size_dim, dtype=np.float64)
    r_train = np.zeros(shape=size_data, dtype=np.float64)
    p_train = np.zeros(shape=size_dim, dtype=np.float64)

    # initial loss
    svm_loss_old = loss_label(x_data, y_data, w_train, b_train, zero_par)

    # train begin
    count_epoch = 0
    timest = time.time()

    # train
    while count_epoch < MAX_EPOCH:

        tmp = np.matmul(w_par2, t_train + one_par - y_data * b_train - mu * r_train) + p_train + z_train / nu
        w_train = np.matmul(w_par1, tmp)
        aw = np.matmul(a_par, w_train)
        b_train = b_par1 * np.dot(b_par2, t_train + one_par - aw - mu * r_train)
        yb = y_data * b_train
        t_train = shrink(aw + yb + mu * r_train + (mu - 1) * one_par, mu)
        z_train = shrink(w_train - nu * p_train, nu * SVMD)

        r_train += (aw + yb - t_train - one_par) / mu
        p_train += (z_train - w_train) / nu

        count_epoch += 1

        if count_epoch % 10 == 0:
            svm_loss = loss_label(x_data, y_data, w_train, b_train, zero_par)
            if SHOW_LOSS:
                print('Training loss for label {0} in {1}: {2}th epoch of {3} - {4}'
                      .format(label_id + 1, label_num, count_epoch, MAX_EPOCH, svm_loss))

            # if meet the accuracy condition
            if np.abs(svm_loss - svm_loss_old) / np.minimum(svm_loss, svm_loss_old) < err_loss:
                break

            svm_loss_old = svm_loss
    # end train

    timeed = time.time()
    print('Training process for label {0} in {1} has finished. Training time: {2} s'
          .format(label_id + 1, label_num, timeed - timest))

    w_train = z_train
    b_train = b_par1 * np.dot(b_par2, t_train + one_par - np.matmul(a_par, w_train) - mu * r_train)

    return w_train, b_train


def prim_tf(x_train, y_train, label_id, label_num):
    # train model for each label
    # method: tensorflow
    # too slow and sometimes could not converge

    print('Training process for label {0} in {1} begin.'.format(label_id + 1, label_num))

    # train data
    x_data = x_train
    y_data = np.zeros(shape=y_train.shape[0], dtype=np.float64)
    y_data[:] = y_train

    # error
    err_loss = 1e-6

    # Ratio of the TRUE data
    print('Ratio of True on train set: {0}, for label {1} in {2}'.format(y_data.mean(), label_id + 1, label_num))

    # Convert labels to +1,-1
    y_data[y_data == 0] = -1

    # data size
    size_data, size_dim = x_data.shape

    # Placeholders
    x = tf.placeholder("float", shape=[None, size_dim])
    y = tf.placeholder("float", shape=None)

    # Define and initialize the network.
    # These are the weights that inform how much each feature contributes to the classification.
    # These weights are available only for this label
    w_train = tf.Variable(tf.zeros([size_dim, 1]))
    b_train = tf.Variable(tf.zeros(shape=()))
    y_raw = tf.matmul(x, w_train) + b_train

    # Loss function
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(w_train))
    # l1_loss = tf.reduce_sum(tf.abs(w_train))
    l1_loss = 0
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([size_data, 1]), 1 - y * y_raw))
    svm_loss = regularization_loss + SVMC * hinge_loss + l1_loss

    # Optimizer
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.05).minimize(svm_loss)
    # train_step = tf.train.FtrlOptimizer(learning_rate=0.05).minimize(svm_loss)

    # Create a local session to run this computation.
    with tf.Session() as sess:
        # Run the initializer to prepare the trainable parameters.
        tf.global_variables_initializer().run()

        # initial loss
        svm_loss_old = sess.run(svm_loss, feed_dict={x: x_data, y: y_data})

        # train begin
        count_epoch = 0
        timest = time.time()

        # train
        while count_epoch < MAX_EPOCH:
            count_epoch += 1
            train_step.run(feed_dict={x: x_data, y: y_data})

            if count_epoch % 100 == 0:
                svm_loss_new = sess.run(svm_loss, feed_dict={x: x_data, y: y_data})
                if SHOW_LOSS:
                    print('Training loss for label {0} in {1}: {2}th epoch of {3} - {4}'
                          .format(label_id + 1, label_num, count_epoch, MAX_EPOCH, svm_loss_new))

                # if meet the accuracy condition
                if np.abs(svm_loss_new - svm_loss_old) / np.minimum(svm_loss_new, svm_loss_old) < err_loss:
                    break

                svm_loss_old = svm_loss_new
        # end train

        timeed = time.time()
        print('Training process for label {0} in {1} has finished. Training time: {2} s'
              .format(label_id + 1, label_num, timeed - timest))

        w_train = sess.run(w_train)
        w_train = w_train[:, 0]
        b_train = sess.run(b_train)

    return w_train, b_train
