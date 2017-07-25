# -*- coding:utf-8 -*-
import pymysql
import progressbar
import time
from ailab.config import *


TEST_MODE = False


def medium_content_with(requestfield):
    # load data from mySQL server

    # make a connection
    conn = pymysql.connect(host=config['host'], user=config['user'], passwd=config['passwd'],
                           db=config['db'], charset=config['charset'])
    print('Successfully connected to the mySQL server.')

    # get a cursor
    cur = conn.cursor()

    # select data from the mySQL server
    timest = time.time()
    cur.execute('select * from medium where {0} is not null'.format(requestfield))
    data0 = cur.fetchall()
    timeed = time.time()
    print('Successfully loaded basic data from the mySQL server. TIME: {0} s'.format(timeed - timest))

    # if test mode
    if TEST_MODE:
        data0 = (data0[0],)

    # obtain articles via article ID
    articlesstr = []

    print('loading articles...')
    time.sleep(0.1)

    bar = progressbar.ProgressBar()
    for i_d in bar(range(len(data0))):

        # request an article
        cur.execute("select content from mediumContent where id='{0}'".format(data0[i_d][0]))
        tmp = cur.fetchall()

        if not tmp == ():  # if exist
            if tmp[0][0] is not None:  # if not empty
                articlesstr.append(tmp[0][0])
            else:
                articlesstr.append('NULL')
        else:
            articlesstr.append('NULL')

    time.sleep(0.1)
    print('Successfully loaded articles from the mySQL server.')

    # close the cursor and the connection
    cur.close()
    conn.close()
    print('mySQL server closed.')

    return data0, articlesstr
