# -*- coding:utf-8 -*-
import pymysql
import progressbar
import time
import config as lib_config


TEST_MODE = False


def medium_content_with(request_str, config=None):
    # load data from mySQL server

    if config is None:
        config = lib_config.config_load

    # make a connection
    try:  # if succeed
        conn = pymysql.connect(host=config['host'], user=config['user'], passwd=config['passwd'],
                               db=config['db'], charset=config['charset'])
        print('Successfully connected to the mySQL server.')

    except pymysql.Error as error:
        # if failed: print error message
        code, message = error.args
        print(code, message)
        print('Connecting to the mySQL server failed. Invalid configuration.')

        # return empty results
        data0 = []
        articlesstr = []
        err = True

        return data0, articlesstr, err

    # get a cursor
    cur = conn.cursor()

    # select data from the mySQL server
    timest = time.time()
    cur.execute('select * from medium where {0} is not null'.format(request_str))
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

    err = False

    return data0, articlesstr, err
