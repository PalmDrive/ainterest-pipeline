# -*- coding:utf-8 -*-
from ailab.train import *
from ailab.data_process import *

# If show results
IF_SHOW_RESULT = True


class MultiClass:

    def multi_classify(articlesstr_in, request, model_dir):

        # request = 'field', 'contentType', 'position'

        is_str = isinstance(articlesstr_in, str)

        if is_str:
            articlesstr = [articlesstr_in]
        else:
            articlesstr = articlesstr_in[:]

        # load dictionaries
        dict_article, dict_label = load_dictionary(request)

        # extract keywords
        articlesjieba = articles_jieba(articlesstr)

        # convert string data to matrix
        x_data = string_to_matrix_article(dict_article, articlesjieba)

        # load trained model
        w = np.loadtxt(model_dir + request + "_w_train.txt")
        b = np.loadtxt(model_dir + request + "_b_train.txt")

        # train results evaluation
        y_pred = predict(x_data, w, b)

        # get labels
        labelsstr = matrix_to_string_label(dict_label, y_pred)

        if is_str:
            labelsstr_out = labelsstr[0]
        else:
            labelsstr_out = labelsstr[:]

        if IF_SHOW_RESULT:
            for i_d in range(len(labelsstr)):
                print(i_d, ':', labelsstr[i_d])

        return labelsstr_out
