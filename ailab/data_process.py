# -*- coding:utf-8 -*-
import ailab.db.db
import numpy as np
import jieba.analyse
import progressbar
import time
import csv
import glob


def field_index_string(requestfield):
    # get the column index corresponding to the field
    field_string_all = ['id', 'mediumType', 'link', 'title', 'summary', 'author',
                        'source', 'keywords', 'labels', 'metaData', 'createdAt',
                        'updatedAt', 'picurl', 'rating', 'ratingsCount', 'viewsCount',
                        'publishedAt', 'algoRating', 'editorRating', 'topics',
                        'featuredPicUrl', 'postedAt', 'group', 'authorId',
                        'editorComment', 'relatedArticles', 'fields', 'positions',
                        'tags', 'hidden', 'labelledField', 'labelledPosition',
                        'labelledPerson', 'labelledCompany', 'labelledContentType',
                        'labelledSentiment', 'simhash', 'contentTypes', 'sentiments',
                        'video', 'duration', 'publishedAtBak']

    # field, position, contentType, sentiment
    field_all = ['id', 'mediumType', 'link', 'title', 'summary', 'author',
                 'source', 'keywords', 'labels', 'metaData', 'createdAt',
                 'updatedAt', 'picurl', 'rating', 'ratingsCount', 'viewsCount',
                 'publishedAt', 'algoRating', 'editorRating', 'topics',
                 'featuredPicUrl', 'postedAt', 'group', 'authorId',
                 'editorComment', 'relatedArticles', 'fields', 'positions',
                 'tags', 'hidden', 'field', 'position',
                 'labelledPerson', 'labelledCompany', 'contentType',
                 'sentiment', 'simhash', 'contentTypes', 'sentiments',
                 'video', 'duration', 'publishedAtBak']

    field_index = field_all.index(requestfield)
    field_str = field_string_all[field_index]

    return field_index, field_str


def get_labels(data0, field_column):

    # get labels for request field
    labelsstr = []

    # invalid label
    invalid_label = ['n', 'u', 'l', '[', ']']
    invalid_len = len(invalid_label)

    for i_d in range(len(data0)):
        tmp = data0[i_d][field_column]

        if tmp == 'null':
            labelsstr.append(['NULL'])
            continue

        labeltmp = eval(tmp)
        if labeltmp == 'null' or labeltmp == '[]':
            labelsstr.append(['NULL'])
            continue

        j_d = 0
        while j_d < invalid_len:
            if invalid_label[j_d] in labeltmp:
                del_index = labeltmp.index(invalid_label[j_d])
                del labeltmp[del_index]
            else:
                j_d += 1

        if len(labeltmp) == 0:
            labelsstr.append(['NULL'])
        else:
            labelsstr.append(labeltmp)

    print('Successfully obtain labels.')

    return labelsstr


def clear_invalid(articlesstr, labelsstr):
    # clear invalid data

    invalid = list()

    for i_d in range(len(labelsstr)):
        if (labelsstr[i_d] == ['NULL']) or (articlesstr[i_d] == 'NULL'):
            invalid.append(i_d)

    invalid = list(set(invalid))
    invalid.sort(reverse=True)
    for i_d in invalid:
        del articlesstr[i_d]
        del labelsstr[i_d]
    return articlesstr, labelsstr


def articles_jieba(articlesstr):
    # extract keywords from articles by using jieba package

    # store extracted keywords and weight from each article
    articlesjieba = list()
    time.sleep(0.1)
    jieba.analyse.extract_tags('Dummy')  # to show that the jieba package are successfully initialized
    time.sleep(0.1)

    # number of keywords for each article
    topkwords = 100

    print('extracting keywords...')
    time.sleep(0.1)

    bar = progressbar.ProgressBar()
    for i_d in bar(range(len(articlesstr))):
        # one article
        article_str = articlesstr[i_d]
        # extract keywords and corresponding weights for this article
        article_jieba = jieba.analyse.extract_tags(article_str, topK=topkwords, withWeight=True)
        # article_jieba = jieba.analyse.textrank(article_str, topK=topkwords, withWeight=True)
        articlesjieba.append(article_jieba)

    time.sleep(0.1)
    print('Successfully extract keywords.')

    return articlesjieba


def build_dictionary(articlesjieba, labelsstr):
    # map keywords to an array index
    dict_article = dict()
    dict_article_len = 0

    # map label words to an array index
    dict_label = dict()
    dict_label_len = 0

    print('building dictionaries...')
    time.sleep(0.1)

    bar = progressbar.ProgressBar()
    for i_d in bar(range(len(labelsstr))):

        # dictionary for labels
        labels = labelsstr[i_d]
        for label_str in labels:
            if not dict_label.__contains__(label_str):
                dict_label[label_str] = dict_label_len
                dict_label_len += 1

        # dictionary for article keywords
        article_jieba = articlesjieba[i_d]
        for wordjieba in article_jieba:
            if not dict_article.__contains__(wordjieba[0]):
                dict_article[wordjieba[0]] = dict_article_len
                dict_article_len += 1

    time.sleep(0.1)
    print('Successfully built dictionaries.')

    return dict_article, dict_label


def string_to_matrix_article(dict_article, articlesjieba):
    # map keywords to float based on dictionaries

    size_dict_article = len(dict_article)
    size_data = len(articlesjieba)

    # each row represents an article, and each column represents a keyword
    articles = np.zeros(shape=(size_data, size_dict_article), dtype=np.float64)

    article = np.zeros(shape=(1, size_dict_article), dtype=np.float64)

    for i_d in range(size_data):
        article[:] = 0
        # wordjieba[0] is the keyword, and wordjieba[1] is the weight
        for wordjieba in articlesjieba[i_d]:
            if dict_article.__contains__(wordjieba[0]):
                # article[0, dict_article[wordjieba[0]]] = wordjieba[1]
                article[0, dict_article[wordjieba[0]]] = 1  # ignore weight
        articles[i_d] = article

    print('Successfully prepared article data.')

    return articles


def string_to_matrix_label(dict_label, labelsstr):
    # map keywords to float based on dictionaries

    size_dict_label = len(dict_label)
    size_data = len(labelsstr)

    # each row represents an article, and each column represents a label
    labels = np.zeros(shape=(size_data, size_dict_label), dtype=np.float64)

    label = np.zeros(shape=(1, size_dict_label), dtype=np.float64)

    for i_d in range(size_data):
        label[:] = 0
        for label_str in labelsstr[i_d]:
            if dict_label.__contains__(label_str):
                label[0, dict_label[label_str]] = 1
        labels[i_d] = label

    print('Successfully prepared label data.')

    return labels


def matrix_to_string_label(dict_label, labels):
    # map keywords to float based on dictionaries

    size_data, size_label = labels.shape

    dict_label_list = list(dict_label.keys())

    # labels
    labelsstr = []

    for i_d in range(size_data):
        label = labels[i_d]
        label_str = []
        for j_d in range(size_label):
            if label[j_d] == 1:
                label_str.append(dict_label_list[j_d])
        labelsstr.append(label_str)

    print('Successfully convert label data from matrix to string.')

    return labelsstr


def divide_data(articles, labels, test_part):
    # divide data set into train and test set

    size_data = articles.shape[0]
    size_test = int(size_data * test_part)
    size_train = size_data - size_test

    # randomly choose some indexes to form a test set, and rest indexes form a train set
    index_all = np.array(range(size_train + size_test))
    index_test = np.random.choice(size_train + size_test, size=size_test, replace=False)
    index_train = np.setdiff1d(index_all, index_test)
    np.random.shuffle(index_train)

    x_train = np.zeros(shape=(size_train, articles.shape[1]), dtype=np.float64)
    y_train = np.zeros(shape=(size_train, labels.shape[1]), dtype=np.float64)
    x_test = np.zeros(shape=(size_test, articles.shape[1]), dtype=np.float64)
    y_test = np.zeros(shape=(size_test, labels.shape[1]), dtype=np.float64)

    x_train[:, :] = articles[index_train, :]
    y_train[:, :] = labels[index_train, :]
    x_test[:, :] = articles[index_test, :]
    y_test[:, :] = labels[index_test, :]

    print('Successfully prepared train and test set.')

    return x_train, y_train, x_test, y_test


def save_dictionary(dict_article, dict_label, requestfield):
    # save dictionaries
    outputdir = "../../output/"
    outputpath = outputdir + requestfield + "/dict_"

    fieldnames = ['keywords', 'index']

    with open(outputpath + "article.txt", "w") as fwright:
        writer = csv.DictWriter(fwright, fieldnames=fieldnames)
        writer.writeheader()
        for key, val in dict_article.items():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: val})

    with open(outputpath + "label.txt", "w") as fwright:
        writer = csv.DictWriter(fwright, fieldnames=fieldnames)
        writer.writeheader()
        for key, val in dict_label.items():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: val})

    print('Successfully saved dictionaries.')


def load_dictionary(requestfield):
    # load dictionaries
    dictdir = "../../output/"
    dictpath = dictdir + requestfield + "/dict_"

    dict_article_path = glob.glob(dictpath + "article.txt")
    dict_label_path = glob.glob(dictpath + "label.txt")

    if len(dict_article_path) == 0:
        print('Article dictionary for {0} was not built yet.'.format(requestfield))
    if len(dict_label_path) == 0:
        print('Label dictionary for {0} was not built yet.'.format(requestfield))
    if len(dict_article_path) == 0 or len(dict_label_path) == 0:
        return dict(), dict()

    fieldnames = ['keywords', 'index']

    dict_article = dict()
    dict_label = dict()

    with open(dictpath + "article.txt", "r") as fread:
        reader = csv.DictReader(fread)
        for row in reader:
            dict_article[row[fieldnames[0]]] = eval(row[fieldnames[1]])

    with open(dictpath + "label.txt", "r") as fread:
        reader = csv.DictReader(fread)
        for row in reader:
            dict_label[row[fieldnames[0]]] = eval(row[fieldnames[1]])

    print('Successfully loaded dictionaries.')

    return dict_article, dict_label


def data_to_libsvm_x(x_data):
    # convert numpy ndarray to the format LIBSVM needed.

    x_data_libsvm = list()

    key0 = list(range(1, x_data[0].shape[0] + 1))

    # x data
    for d in x_data:
        x_data_libsvm.append(dict(zip(key0, d.tolist())))

    return x_data_libsvm


def data_to_libsvm_y(y_data):
    # convert numpy ndarray to the format LIBSVM needed.

    if len(y_data.shape) == 2:

        y_data_libsvm = []
        y_data_column = np.zeros(shape=y_data.shape[0], dtype=float)

        for i_d in range(y_data.shape[1]):
            np.copyto(y_data_column, y_data[:, i_d])
            y_data_libsvm.append(y_data_column.tolist())
    else:
        y_data_libsvm = y_data.tolist()

    return y_data_libsvm


def get_data(requestfield='field', config=None, test_part=0.1):

    # index for request field
    data_index, field_str = field_index_string(requestfield)

    # field string

    # request articles and labels (format: string)
    data0, articlesstr, err_mysql = ailab.db.db.medium_content_with(field_str, config)

    # if error:
    if err_mysql:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        err = True

        return x_train, y_train, x_test, y_test, err

    # get labels for request field (format: string)
    labelsstr = get_labels(data0, data_index)

    # clear invalid data
    articlesstr, labelsstr = clear_invalid(articlesstr, labelsstr)

    # extract keywords from articles
    articlesjieba = articles_jieba(articlesstr)

    # build dictionaries so that keywords and labels could be mapped to a float number
    dict_article, dict_label = build_dictionary(articlesjieba, labelsstr)

    # save data
    save_dictionary(dict_article, dict_label, requestfield)

    # convert string data to float matrix
    articles = string_to_matrix_article(dict_article, articlesjieba)
    labels = string_to_matrix_label(dict_label, labelsstr)

    # divide data set into train and test set
    x_train, y_train, x_test, y_test = divide_data(articles, labels, test_part)

    err = False

    return x_train, y_train, x_test, y_test, err


def get_data_test(trainnum=10000, testnum=1000, datadim=100, labelnum=2):
    # a function for testing the train model

    # train data
    x_train = np.random.random([trainnum, datadim]) * 2 - 1
    x_train = np.sign(x_train)
    y_train = np.zeros(shape=(trainnum, labelnum), dtype=np.float64)
    tmp = np.sign(x_train[:, 0])
    tmp[tmp == -1] = 0
    y_train[:, 0] = tmp
    tmp = np.sign(x_train[:, -1])
    tmp[tmp == -1] = 0
    y_train[:, -1] = tmp

    # test data
    x_test = np.random.random([testnum, datadim]) * 2 - 1
    x_test = np.sign(x_test)
    y_test = np.zeros(shape=(testnum, labelnum), dtype=np.float64)
    tmp = np.sign(x_test[:, 0])
    tmp[tmp == -1] = 0
    y_test[:, 0] = tmp
    tmp = np.sign(x_test[:, -1])
    tmp[tmp == -1] = 0
    y_test[:, -1] = tmp

    err = False

    return x_train, y_train, x_test, y_test, err


def get_data_file():
    # load data from a file
    # unfinished

    articlesstr = ['这是一篇示例文章', '还有一篇文章，测试', '当然是选择原谅她啦', '今天你要嫁给我']

    return articlesstr
