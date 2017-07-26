# -*- coding:utf-8 -*-
from ailab.train import *
from ailab.data_process import *

# If save data
IF_SAVE = True

# If show results
IF_SHOW_RESULT = False

# If plot
IF_PLOT = True


def main():
    # requested
    request = 'contentType'

    # Get data
    x_train, y_train, x_test, y_test = get_data(request)

    # train
    w, b = train_field(x_train, y_train, thread=8)

    if IF_SAVE:
        np.savetxt("../output/" + request + "_w_train.txt", w)
        np.savetxt("../output/" + request + "_b_train.txt", b)

    # train results evaluation
    y_train_pred = predict(x_train, w, b)
    y_test_pred = predict(x_test, w, b)
    print("Accuracy on train:", accuracy(y_train, y_train_pred))
    print("Accuracy on test:", accuracy(y_test, y_test_pred))

    if IF_SHOW_RESULT:
        print('w:\n', w)
        print('b:\n', b)
        print('norm(w):\n', np.linalg.norm(w, ord=2, axis=0))

    if IF_PLOT:
        data_plot(y_train, y_test, y_train_pred, y_test_pred)


if __name__ == '__main__':
    main()
