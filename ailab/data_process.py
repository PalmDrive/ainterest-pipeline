# -*- coding:utf-8 -*-
from request_mysql import request_mysql
import numpy as np
import jieba.analyse
import progressbar
import time


def field_index(requestfield):
    # get the column index corresponding to the field
    # all available fields
    field_all = ['id', 'mediumType', 'link', 'title', 'summary', 'author',
                 'source', 'keywords', 'labels', 'metaData', 'createdAt',
                 'updatedAt', 'picurl', 'rating', 'ratingsCount', 'viewsCount',
                 'publishedAt', 'algoRating', 'editorRating', 'topics',
                 'featuredPicUrl', 'postedAt', 'group', 'authorId',
                 'editorComment', 'relatedArticles', 'fields', 'positions',
                 'tags', 'hidden', 'labelledField', 'labelledPosition',
                 'labelledPerson', 'labelledCompany', 'labelledContentType',
                 'labelledSentiment', 'simhash', 'contentTypes', 'sentiments',
                 'video', 'duration', 'publishedAtBak']
    return field_all.index(requestfield)


def get_labels(data0, field_column):

    # get labels for request field
    labelsstr = []

    for i_d in range(len(data0)):
        tmp = data0[i_d][field_column]
        if tmp == 'null':
            labelsstr.append(['NULL'])
        elif tmp == '[]':
            labelsstr.append(['NULL'])
        else:
            labelsstr.append(eval(tmp))
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


def build_dictionary(articlesstr, labelsstr):
    # map keywords to an array index
    dict_article = dict()
    dict_article_len = 0

    # map label words to an array index
    dict_label = dict()
    dict_label_len = 0

    # store extracted keywords and weight from each article
    # jieba package are used
    articlesjieba = list()
    time.sleep(0.1)
    jieba.analyse.extract_tags('Dummy')  # to show that the jieba package are successfully initialized
    time.sleep(0.1)

    # number of keywords for each article
    topkwords = 100

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
        article_str = articlesstr[i_d]
        # extract keywords and corresponding weights for each article
        article_jieba = jieba.analyse.extract_tags(article_str, topK=topkwords, withWeight=True)
        for wordjieba in article_jieba:
            if not dict_article.__contains__(wordjieba[0]):
                dict_article[wordjieba[0]] = dict_article_len
                dict_article_len += 1
        articlesjieba.append(article_jieba)

    time.sleep(0.1)
    print('Successfully built dictionaries.')

    return dict_article, dict_label, articlesjieba


def prepare_data(dict_article, dict_label, articlesjieba, labelsstr):
    # map keywords to float based on dictionaries

    size_dict_article = len(dict_article)
    size_dict_label = len(dict_label)
    size_data = len(labelsstr)

    # each row represents an article, and each column represents a keyword
    # data set contains keyword weights for each article
    # zero means that this keyword is not shown for this article
    articles = np.zeros(shape=(size_data, size_dict_article), dtype=np.float64)

    # each row represents an article, and each column represents a label
    # label set contains true labels for each article
    # one (zero) means that this article does (not) belong to this label
    labels = np.zeros(shape=(size_data, size_dict_label), dtype=np.float64)

    article = np.zeros(shape=(1, size_dict_article), dtype=np.float64)
    label = np.zeros(shape=(1, size_dict_label), dtype=np.float64)

    for i_d in range(size_data):

        article[:] = 0
        for wordjieba in articlesjieba[i_d]:
            # wordjieba[0] is the keyword, and wordjieba[1] is the weight
            # article[0, dict_article[wordjieba[0]]] = wordjieba[1]
            article[0, dict_article[wordjieba[0]]] = 1  # ignore weight
        articles[i_d] = article

        label[:] = 0
        for label_str in labelsstr[i_d]:
            label[0, dict_label[label_str]] = 1
        labels[i_d] = label

    print('Successfully prepared data set.')

    return articles, labels


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


def get_data(requestfield='labelledField', test_mode=False, test_part=0.1):

    # index for request field
    data_index = field_index(requestfield)

    # request articles and labels (format: string)
    data0, articlesstr = request_mysql(requestfield, test_mode)

    # get labels for request field (format: string)
    labelsstr = get_labels(data0, data_index)

    # clear invalid data
    articlesstr, labelsstr = clear_invalid(articlesstr, labelsstr)

    # build dictionaries so that keywords and labels could be mapped to a float number
    dict_article, dict_label, articlesjieba = build_dictionary(articlesstr, labelsstr)

    # convert string data to float matrix
    # the data set
    articles, labels = prepare_data(dict_article, dict_label, articlesjieba, labelsstr)

    # divide data set into train and test set
    x_train, y_train, x_test, y_test = divide_data(articles, labels, test_part)

    return x_train, y_train, x_test, y_test


def get_data_test(trainnum=10000, testnum=1000, datadim=100):
    # a function for testing the train model

    # train data
    x_train = np.random.random([trainnum, datadim]) * 2 - 1
    x_train = np.sign(x_train)
    y_train = np.zeros(shape=(trainnum, 2), dtype=np.float64)
    tmp = np.sign(x_train[:, 0])
    tmp[tmp == -1] = 0
    y_train[:, 0] = tmp
    tmp = np.sign(x_train[:, -1])
    tmp[tmp == -1] = 0
    y_train[:, 1] = tmp

    # test data
    x_test = np.random.random([testnum, datadim]) * 2 - 1
    x_test = np.sign(x_test)
    y_test = np.zeros(shape=(testnum, 2), dtype=np.float64)
    tmp = np.sign(x_test[:, 0])
    tmp[tmp == -1] = 0
    y_test[:, 0] = tmp
    tmp = np.sign(x_test[:, -1])
    tmp[tmp == -1] = 0
    y_test[:, 1] = tmp

    return x_train, y_train, x_test, y_test


def get_data_file():
    # load data from a file

    with open('labelledField_dict_article.txt', 'r') as read_file:
        tmp = read_file.read()
    dict_article = eval(tmp)
    with open('labelledField_dict_label.txt', 'r') as read_file:
        tmp = read_file.read()
    dict_label = eval(tmp)
    with open('labelledField_articlesjieba.txt', 'r') as read_file:
        tmp = read_file.read()
    articlesjieba = eval(tmp)
    with open('labelledField_labelsstr.txt', 'r') as read_file:
        tmp = read_file.read()

    labelsstr = eval(tmp)

    articles, labels = prepare_data(dict_article, dict_label, articlesjieba, labelsstr)
    x_train, y_train, x_test, y_test = divide_data(articles, labels, test_part=0.1)

    return x_train, y_train, x_test, y_test
