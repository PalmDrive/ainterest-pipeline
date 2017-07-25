# -*- coding:utf-8 -*-
from ailab.train import *
from ailab.data_process import *

# If plot
IF_PLOT = True


def main():
    # Get the data.
    # x_train, y_train, x_test, y_test = get_data('labelledPosition')
    # x_train, y_train, x_test, y_test = get_data('labelledField')
    # x_train, y_train, x_test, y_test = get_data('labelledContentType')
    x_train, y_train, x_test, y_test = get_data_test(200, 20, 2000)  # very simple test data
    # x_train, y_train, x_test, y_test = get_data_file()

    w, b = train_field(x_train, y_train, thread=4)
    y_train_pred = predict(x_train, w, b)
    y_test_pred = predict(x_test, w, b)

    print('x_train:\n', x_train)
    print('y_train:\n', y_train)

    print('w:\n', w)
    print('b:\n', b)
    print('norm(w):\n', np.linalg.norm(w, ord=2, axis=0))

    print("Accuracy on train:", accuracy(y_train, y_train_pred))
    print("Accuracy on test:", accuracy(y_test, y_test_pred))

    if IF_PLOT:
        data_plot(y_train, y_test, y_train_pred, y_test_pred)


if __name__ == '__main__':
    main()
