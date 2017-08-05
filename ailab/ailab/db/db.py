# -*- coding:utf-8 -*-
import pymysql
import progressbar
import time
import yaml
# from ailab.config import config_load


TEST_MODE = False


class DB:

    def __init__(self, config=None):

        if config is None:
            with open('../../../config/config.yaml', 'r') as read_file:
                self.__config = yaml.load(read_file)
        else:
            self.__config = config

    def set_config(self, config=None):

        if config is None:
            with open('../../../config/config.yaml', 'r') as read_file:
                self.__config = yaml.load(read_file)
        else:
            self.__config = config

    def test_connect(self):
        config = self.__config

        if config is None:
            # if no configuration
            print('No configure setting.')

            err = True

            return err

        # try to make a connection
        try:  # if succeed
            conn = pymysql.connect(host=config['host'], user=config['user'],
                                   passwd=config['passwd'], db=config['db'])
            conn.close()

            # announcement
            print('Connection is OK.')

            err = False

        except pymysql.Error as error:
            # if failed: print error message
            code, message = error.args

            # announcement
            print(code, message)
            print('Connection is Invalid.')

            err = True

        return err

    def medium_content_with(self, request_str):
        # load data from mySQL server

        # configuration
        config = self.__config

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
            articles_str = []
            err = True

            return data0, articles_str, err

        # get a cursor
        cur = conn.cursor()

        # announcement
        print('Begin loading data...')

        # begin a timer
        time_st = time.time()

        # select data from the mySQL server
        cur.execute('select * from medium where {0} is not null'.format(request_str))

        # fetch data
        data0 = cur.fetchall()

        # end the timer
        time_ed = time.time()

        print('Successfully loaded basic data from the mySQL server. TIME: {0} s'.format(time_ed - time_st))

        # if test mode
        if TEST_MODE:
            data0 = (data0[0],)

        # obtain articles via article ID
        articles_str = []

        print('loading articles...')
        time.sleep(0.1)

        # begin a progress bar
        bar = progressbar.ProgressBar()

        # load articles
        for i_d in bar(range(len(data0))):

            # request an article
            cur.execute("select content from mediumContent where id='{0}'".format(data0[i_d][0]))

            # fetch this article
            tmp = cur.fetchall()

            if not tmp == ():  # if exist
                if tmp[0][0] is not None:  # if not empty
                    articles_str.append(tmp[0][0])
                else:
                    articles_str.append('NULL')
            else:
                articles_str.append('NULL')

        time.sleep(0.1)
        print('Successfully loaded articles from the mySQL server.')

        # close the cursor and the connection
        cur.close()
        conn.close()
        print('mySQL server closed.')

        # now everything is OK
        err = False

        return data0, articles_str, err
