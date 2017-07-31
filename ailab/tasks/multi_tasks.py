# -*- coding:utf-8 -*-
from ailab.train import *
from ailab.data_process import *


# If save data
IF_SAVE = False

# If show results
IF_SHOW_RESULT = True

# If plot
IF_PLOT = True


class MultiClass:

    # If show results
    if_show_result = IF_SHOW_RESULT

    def multi_classify(self, articlesstr_in, request, model_dir):

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

        if self.if_show_result:
            for i_d in range(len(labelsstr)):
                print(i_d, ':', labelsstr[i_d])

        return labelsstr_out


class MultiTrain:

    # If save data
    if_save = IF_SAVE

    # If show results
    if_show_result = IF_SHOW_RESULT

    # If plot
    if_plot = IF_PLOT

    def multi_train(self, request, output_dir):

        # request = 'field', 'contentType', 'position'

        # Get data
        x_train, y_train, x_test, y_test = get_data(request)

        # Or:
        # x_train, y_train, x_test, y_test = get_data_test(200, 20, 1000)  # very simple test data
        # x_train, y_train, x_test, y_test = get_data_file()

        # train
        w, b = train_field(x_train, y_train, thread=6)

        if self.if_save:
            np.savetxt(output_dir + request + "_w_train.txt", w)
            np.savetxt(output_dir + request + "_b_train.txt", b)

        # train results evaluation
        y_train_pred = predict(x_train, w, b)
        y_test_pred = predict(x_test, w, b)
        print("Accuracy on train:", accuracy(y_train, y_train_pred))
        print("Accuracy on test:", accuracy(y_test, y_test_pred))

        if self.if_show_result:
            print('w:\n', w)
            print('b:\n', b)
            print('norm(w):\n', np.linalg.norm(w, ord=2, axis=0))

        if self.if_plot:
            data_plot(y_train, y_test, y_train_pred, y_test_pred)
