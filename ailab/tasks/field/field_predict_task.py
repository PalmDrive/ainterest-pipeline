# -*- coding:utf-8 -*-
from ailab.train import *
from ailab.data_process import *

# If show results
IF_SHOW_RESULT = False


def main():
    # requested
    request = 'field'

    # Get data
    articlesstr = get_data_file()

    # load dictionaries
    dict_article, dict_label = load_dictionary(request)

    # extract keywords
    articlesjieba = articles_jieba(articlesstr)

    # convert string data to matrix
    x_data = string_to_matrix_article(dict_article, articlesjieba)

    # load trained model
    w = np.loadtxt("../output/" + request + "_w_train.txt")
    b = np.loadtxt("../output/" + request + "_b_train.txt")

    # train results evaluation
    y_pred = predict(x_data, w, b)

    # get labels
    labelsstr = matrix_to_string_label(dict_label, y_pred)

    if IF_SHOW_RESULT:
        for i_d in range(len(labelsstr)):
            print(i_d, ':', labelsstr[i_d])


if __name__ == '__main__':
    main()
