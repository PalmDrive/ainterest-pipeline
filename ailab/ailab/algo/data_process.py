# -*- coding:utf-8 -*-
from ailab.db.db import DB
import numpy as np
import jieba.analyse as ja
import progressbar
import time
import csv
import glob


def field_index_string(request_field):
    # get the column index corresponding to the field
    field_string_all = ['id', 'mediumType', 'link', 'title', 'summary',
                        'author',
                        'source', 'keywords', 'labels', 'metaData', 'createdAt',
                        'updatedAt', 'picurl', 'rating', 'ratingsCount',
                        'viewsCount',
                        'publishedAt', 'algoRating', 'editorRating', 'topics',
                        'featuredPicUrl', 'postedAt', 'group', 'authorId',
                        'editorComment', 'relatedArticles', 'fields',
                        'positions',
                        'tags', 'hidden', 'labelledField', 'labelledPosition',
                        'labelledPerson', 'labelledCompany',
                        'labelledContentType',
                        'labelledSentiment', 'simhash', 'contentTypes',
                        'sentiments',
                        'video', 'duration', 'publishedAtBak']

    # field, position, content_type, sentiment
    field_all = ['id', 'mediumType', 'link', 'title', 'summary', 'author',
                 'source', 'keywords', 'labels', 'metaData', 'createdAt',
                 'updatedAt', 'picurl', 'rating', 'ratingsCount', 'viewsCount',
                 'publishedAt', 'algoRating', 'editorRating', 'topics',
                 'featuredPicUrl', 'postedAt', 'group', 'authorId',
                 'editorComment', 'relatedArticles', 'fields', 'positions',
                 'tags', 'hidden', 'field', 'position',
                 'labelledPerson', 'labelledCompany', 'content_type',
                 'sentiment', 'simhash', 'contentTypes', 'sentiments',
                 'video', 'duration', 'publishedAtBak']

    field_index = field_all.index(request_field)
    field_str = field_string_all[field_index]

    return field_index, field_str


def get_labels(data0, field_column):
    # get labels for request field
    labels_str = []

    # invalid label
    invalid_label = ['n', 'u', 'l', '[', ']']

    # get labels
    for i_d in range(len(data0)):
        # get the data as a string format
        tmp = data0[i_d][field_column]

        # ignore if is 'null'
        if tmp == 'null':
            labels_str.append(['NULL'])
            continue

        # then get the data
        label_tmp = eval(tmp)

        # ignore if is 'null' or '[]'
        if label_tmp == 'null' or label_tmp == '[]':
            labels_str.append(['NULL'])
            continue

        # ignore invalid data
        for invalid_d in invalid_label:
            while invalid_d in label_tmp:
                del label_tmp[label_tmp.index(invalid_d)]

        # ignore if is empty
        if len(label_tmp) == 0:
            labels_str.append(['NULL'])
            continue

        # now everything is OK
        labels_str.append(label_tmp)

    # finish
    print('Successfully obtained labels.')

    return labels_str


def clear_invalid(articles_str, labels_str):
    # clear invalid data

    # first find them
    invalid = list()

    # find invalid indexes
    for i_d in range(len(labels_str)):
        if (labels_str[i_d] == ['NULL']) or (articles_str[i_d] == 'NULL'):
            invalid.append(i_d)

    # remove duplication
    invalid = list(set(invalid))
    invalid.sort(reverse=True)

    # second delete them
    for i_d in invalid:
        del articles_str[i_d]
        del labels_str[i_d]

    return articles_str, labels_str


def articles_to_jieba(articles_str):
    # extract keywords from articles by using jieba package

    # store extracted keywords and weight from each article
    articles_jieba = list()

    # to show that the jieba package are successfully initialized
    time.sleep(0.1)
    ja.extract_tags('Dummy')
    time.sleep(0.1)

    # number of keywords for each article
    top_k = 100

    # announcement
    print('extracting keywords...')
    time.sleep(0.1)

    # begin a progressbar
    bar = progressbar.ProgressBar()

    # for each article
    for article in bar(articles_str):
        # extract keywords and corresponding weights for this article
        tmp = ja.extract_tags(article, topK=top_k, withWeight=True)
        # tmp = ja.textrank(article, topK=top_k, withWeight=True)

        # convert unicode to string
        tmp_str = []
        for tmp_i in tmp:
            tmp_str.append((tmp_i[0].encode('utf-8'), tmp_i[1]))

        # append
        articles_jieba.append(tmp_str)

    # finish
    time.sleep(0.1)
    print('Successfully extract keywords.')

    return articles_jieba


def build_dictionary(articles_jieba, labels_str):
    # map article keywords to an integer
    dict_article = dict()
    dict_article_len = 0

    # map label words to an integer
    dict_label = dict()
    dict_label_len = 0

    # begin
    print('building dictionaries...')
    time.sleep(0.1)

    # begin a progressbar
    bar = progressbar.ProgressBar()

    # for each
    for i_d in bar(range(len(labels_str))):

        # dictionary for labels
        label_list = labels_str[i_d]
        for label in label_list:
            if not dict_label.__contains__(label):
                dict_label[label] = dict_label_len
                dict_label_len += 1

        # dictionary for article keywords
        article_keywords = articles_jieba[i_d]
        for word in article_keywords:
            if not dict_article.__contains__(word[0]):
                dict_article[word[0]] = dict_article_len
                dict_article_len += 1

    # finish
    time.sleep(0.1)
    print('Successfully built dictionaries.')

    return dict_article, dict_label


def string_to_matrix_article(dict_article, articles_jieba):
    # map keywords to float based on dictionaries

    size_dict_article = len(dict_article)
    size_data = len(articles_jieba)

    # each row represents an article, and each column represents a keyword
    articles = np.zeros(shape=(size_data, size_dict_article), dtype=np.float64)

    # for each article
    for i_d in range(size_data):
        # word[0] is the keyword, and word[1] is the weight
        for word in articles_jieba[i_d]:
            if dict_article.__contains__(word[0]):
                # articles[i_d, dict_article[word[0]]] = word[1]
                articles[i_d, dict_article[word[0]]] = 1  # ignore weight

    # finish
    print('Successfully prepared article data.')

    return articles


def string_to_matrix_label(dict_label, labels_str):
    # map label string to float based on dictionaries

    size_dict_label = len(dict_label)
    size_data = len(labels_str)

    # each row represents an article, and each column represents a label
    labels = np.zeros(shape=(size_data, size_dict_label), dtype=np.float64)

    for i_d in range(size_data):
        for label_str in labels_str[i_d]:
            if dict_label.__contains__(label_str):
                labels[i_d, dict_label[label_str]] = 1

    # finish
    print('Successfully prepared label data.')

    return labels


def matrix_to_string_label(dict_label, labels):
    # map keywords to float based on dictionaries

    # size
    s_d, s_l = labels.shape

    # dictionary to a list
    d_keys = list(dict_label.keys())
    d_values = list(dict_label.values())

    # labels as string
    labels_str = []

    # for each record
    for i_d in range(s_d):
        tmp = labels[i_d]
        label_tmp = [d_keys[d_values.index(j_d)] for j_d in range(s_l) if tmp[j_d] == 1]
        labels_str.append(label_tmp)

    # finish
    print('Successfully convert label data from matrix to string.')

    return labels_str


def divide_data(articles, labels, test_part):
    # divide data into train and test set

    # size
    s_d, size_article = articles.shape
    size_label = labels.shape[1]
    s_te = int(s_d * test_part)
    s_tr = s_d - s_te

    # randomly choose some indexes to form a test set, and rest indexes form a train set
    index_all = np.array(range(s_d))
    index_test = np.random.choice(s_d, size=s_te, replace=False)
    index_train = np.setdiff1d(index_all, index_test)
    np.random.shuffle(index_train)

    # data format: numpy ndarray
    x_train = np.zeros(shape=(s_tr, size_article), dtype=np.float64)
    y_train = np.zeros(shape=(s_tr, size_label), dtype=np.float64)
    x_test = np.zeros(shape=(s_te, size_article), dtype=np.float64)
    y_test = np.zeros(shape=(s_te, size_label), dtype=np.float64)

    # train and test data
    x_train[:, :] = articles[index_train, :]
    y_train[:, :] = labels[index_train, :]
    x_test[:, :] = articles[index_test, :]
    y_test[:, :] = labels[index_test, :]

    # finish
    print('Successfully prepared train and test set.')

    return x_train, y_train, x_test, y_test


def save_dictionary(dict_article, dict_label, output_dir):
    # save dictionaries
    output_path = output_dir + "/dict_"

    # field name
    fieldnames = ['keywords', 'index']

    # open and write data
    with open(output_path + 'article.txt', 'w') as f_wright:
        # csv dict writer
        writer = csv.DictWriter(f_wright, fieldnames=fieldnames)
        writer.writeheader()
        for key, val in dict_article.items():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: val})

    # open and write data
    with open(output_path + 'label.txt', 'w') as f_wright:
        # csv dict writer
        writer = csv.DictWriter(f_wright, fieldnames=fieldnames)
        writer.writeheader()
        for key, val in dict_label.items():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: val})

    # finish
    print('Successfully saved dictionaries.')


def load_dictionary(dict_dir):
    # load dictionaries
    dict_path = dict_dir + "/dict_"

    # find them
    dict_article_path = glob.glob(dict_path + "article.txt")
    dict_label_path = glob.glob(dict_path + "label.txt")

    # if could not find the directory for articles
    if len(dict_article_path) == 0:
        print('Article dictionary was not found here:')
        print('\t' + dict_dir)

    # if could not find the directory for labels
    if len(dict_label_path) == 0:
        print('Label dictionary was not found here:')
        print('\t' + dict_dir)

    # if either does not exist
    if len(dict_article_path) == 0 or len(dict_label_path) == 0:
        return dict(), dict()

    # field names
    fieldnames = ['keywords', 'index']

    # first empty list
    dict_article = dict()
    dict_label = dict()

    # open and read data
    with open(dict_path + 'article.txt', 'r') as f_read:
        # csv dict reader
        reader = csv.DictReader(f_read)
        for row in reader:
            dict_article[row[fieldnames[0]]] = eval(row[fieldnames[1]])

    # open and read data
    with open(dict_path + 'label.txt', 'r') as f_read:
        # csv dict reader
        reader = csv.DictReader(f_read)
        for row in reader:
            dict_label[row[fieldnames[0]]] = eval(row[fieldnames[1]])

    # finish
    print('Successfully loaded dictionaries.')

    return dict_article, dict_label


def data_to_libsvm_x(x_data):
    # convert numpy ndarray to the format LIBSVM needed.

    x_data_libsvm = list()

    # x data, sparse
    for d in x_data:
        items = dict()
        for i_d in range(len(d)):
            if d[i_d] != 0:
                items[i_d + 1] = d[i_d]
        x_data_libsvm.append(items)

    return x_data_libsvm


def data_to_libsvm_y(y_data):
    # convert numpy array to the format LIBSVM needed.

    # if y_data is a two dimensional array
    if len(y_data.shape) == 2:

        # list beyond list
        y_data_libsvm = []

        # each column for original numpy data
        y_data_column = np.zeros(shape=y_data.shape[0], dtype=float)

        # numpy array to list
        for i_d in range(y_data.shape[1]):
            np.copyto(y_data_column, y_data[:, i_d])
            y_data_libsvm.append(y_data_column.tolist())
    else:  # if y_data is a one-dimensional array
        y_data_libsvm = y_data.tolist()

    return y_data_libsvm


def get_data(request_field='field', config=None, test_part=0.1, output_dir='../../../output'):
    # index and string for requested field
    data_index, field_str = field_index_string(request_field)

    # database
    db = DB(config)

    # request articles and labels (format: string)
    data0, articles_str, err_mysql = db.medium_content_with(field_str)

    # if error:
    if err_mysql:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        err = True

        return x_train, y_train, x_test, y_test, err

    # get labels for request field (format: string)
    labels_str = get_labels(data0, data_index)

    # clear invalid data
    articles_str, labels_str = clear_invalid(articles_str, labels_str)

    # extract keywords from articles
    articles_jieba = articles_to_jieba(articles_str)

    # build dictionaries so that keywords and labels could be mapped to a float number
    dict_article, dict_label = build_dictionary(articles_jieba, labels_str)

    # save data
    save_dictionary(dict_article, dict_label, output_dir)

    # convert string data to float matrix
    articles = string_to_matrix_article(dict_article, articles_jieba)
    labels = string_to_matrix_label(dict_label, labels_str)

    # divide data set into train and test set
    x_train, y_train, x_test, y_test = divide_data(articles, labels, test_part)

    # now everything is OK
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

    articles_str = ['这是一篇示例文章', '还有一篇文章，测试']

    return articles_str
