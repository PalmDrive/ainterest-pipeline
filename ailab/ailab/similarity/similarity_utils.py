# -*- coding:utf-8 -*-
import jieba
import jieba.analyse as jieba_analyse
import time
from ailab.db.db import DB
from multiprocessing import Pool as ThreadPool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def test_connect(config=None):

    # database
    db = DB(config)

    # set
    config = db.set_config(config)

    # test the connection via class DB
    connect_err = db.test_connect()

    return config, connect_err


def unicode_to_str(uni_list):
    if not isinstance(uni_list, list):
        return uni_list.encode('utf-8')

    str_list = list()

    for d in uni_list:
        str_list.append(d.encode('utf-8'))

    return str_list


def recent_articles(config, total_num=None):

    # database
    db = DB(config)

    # get raw articles and ids
    articles_raw, id_list, err = db.recent_articles(number=total_num)

    # convert to UTF-8
    articles_raw = unicode_to_str(articles_raw)
    id_list = unicode_to_str(id_list)

    # if error
    if err is True:
        return [], []

    return articles_raw, id_list


def jieba_cut_article(article_raw):

    # invalid string list
    invalid_str = [u' ', u';', u',', u'.', u"'", u'"', u'-',
                   u'；', u'，', u'。', u'‘', u'’', u'“', u'”', u'—',
                   u'!', u'！', u'?', u'？', u'@', u'、', u'\t',
                   u':', u'：', u'/', u'\\', u'\r', u'…', u'(', u')',
                   u'（', u'）', u'{', u'}', u'[', u']', u'|',
                   u'+', u'=', u'<', u'>', u'《', u'》']

    # if is NULL
    if article_raw == 'NULL':
        return []

    # remove whitespace at two ends
    article_raw_new = article_raw.strip()

    # jieba cut to get Chinese words
    article_jieba = jieba.cut(article_raw_new)

    # remove whitespace, and encoding
    jieba_list = ' '.join(article_jieba).split()

    # remove invalid string
    article = [d for d in jieba_list if d not in invalid_str]

    return article


def jieba_cut(articles_raw, thread=8):

    # for a raw article string
    if not isinstance(articles_raw, list):
        return jieba_cut_article(articles_raw)

    # prepare jieba cache for multi-thread running
    jieba_analyse.extract_tags('Dummy')
    time.sleep(0.1)

    # multi-thread
    print('Multi-thread: ' + str(thread))

    # thread pool
    pool = ThreadPool(thread)

    # train each label
    results = pool.map(jieba_cut_article, articles_raw)

    # close the pool
    pool.close()

    # re-arrange the results
    articles_list = [article for article in results]

    return articles_list


def auto_tags(id_list=None, id_list_old=None, extend=0):

    # automatically create tags

    # if old id list is empty
    if id_list_old is None:

        # if new id list is empty
        if id_list is None:
            len_id_list = 0
        else:
            len_id_list = len(id_list)

        # this situation happened mainly when initializing a model
        tags_list = list(range(len_id_list + extend))

        return tags_list

    # current tag for new id
    count = len(id_list_old)

    # no new id
    if id_list is None:

        # this situation is unlikely to happen
        tags_list = list(range(count, count + extend))

        return tags_list

    # tags list
    tags_list = list()

    # for each new id
    for d in id_list:

        # if contained in old id list
        if d in id_list_old:
            tags_list.append(id_list_old.index(d))
        else:  # for new id
            tags_list.append(count)
            count += 1

    # if no extra tags
    if extend == 0:
        return tags_list

    # add some extra tags, and this situation is not about to happen
    tags_extend = list(range(count, count + extend))
    tags_list.extend(tags_extend)

    return tags_list


def merge_id(id_list, id_list_old):

    # merge new id list with the old one.

    # if old id list is empty
    if id_list_old is None:
        return id_list

    # the merged list
    id_list_merge = list()

    # old id list
    for d in id_list_old:
        id_list_merge.append(d)

    # new id list
    for d in id_list:
        if d not in id_list_old:
            id_list_merge.append(d)

    return id_list_merge


def tagged_docs(articles_list, tags_list):

    # for empty input
    if len(tags_list) == 0:
        print('No article!')
        return []

    # for only one article
    if not isinstance(tags_list, list):

        # it is only an article
        if isinstance(articles_list[0], list):
            article = articles_list[0]
        else:
            article = articles_list

        # tagged document
        doc = TaggedDocument(article, [tags_list])

        return doc

    # for many articles
    docs = list()

    # if need extra articles
    len_diff = len(tags_list) - len(articles_list)

    # a new articles_list
    articles_list_extend = list()

    # copy
    for d in articles_list:
        articles_list_extend.append(d)

    # append empty articles
    for d in range(len_diff):
        articles_list_extend.append([])

    # for each article
    for i_d in range(len(tags_list)):

        # article
        article = articles_list_extend[i_d]

        # article id
        tag_index = tags_list[i_d]

        # tagged document
        doc = TaggedDocument(article, [tag_index])

        # tagged documents
        docs.append(doc)

    return docs


def most_similar_articles(sim_articles, id_list, thres):

    # if sim_articles is just one article with a weight
    if not isinstance(sim_articles, list):

        # for this case, return None
        if sim_articles[1] < thres:
            return None

        # get article id
        id_str = id_list[sim_articles[0]]

        # article
        most_sim_article = (id_str, sim_articles[1])

        return most_sim_article

    # most similar articles whose weights are beyond the threshold
    most_sim_articles = list()

    # judge each article
    for article in sim_articles:
        if article[1] < thres:
            break

        # get article id
        id_str = id_list[article[0]]

        most_sim_articles.append((id_str, article[1]))

    return most_sim_articles
