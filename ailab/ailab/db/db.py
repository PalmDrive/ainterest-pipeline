# -*- coding:utf-8 -*-
import pymysql
import progressbar
import time
import yaml
import os


TEST_MODE = False


class DB:

    def __init__(self, config=None):

        self.__config = None

        self.set_config(config)

    def set_config(self, config=None):

        # if empty or None: do nothing
        if config is None:
            return self.__config

        # if config is a path or a directory
        if not isinstance(config, dict):

            # if is dir
            if os.path.isdir(config):
                config_path = os.path.join(config, 'config.yaml')
            else:
                config_path = config

            # if file does not exist
            if not os.path.isfile(config_path):
                print('Config file does not exist here:')
                print('\t' + config_path)
                return self.__config

            # load config from file
            with open(config_path, 'r') as read_file:
                config = yaml.load(read_file)

        # old config
        config_old = self.__config

        # if old config is empty
        if config_old is None:
            self.__config = config
            return self.__config

        # users might set only part of the configuration
        for key in config:
            config_old[key] = config[key]
        config = config_old

        # set
        self.__config = config

        return self.__config

    def test_connect(self):

        # exam the format of config
        config = self.__config

        if config is None:
            # if no configuration
            print('No configure setting.')

            err = True

            return err

        # if all keywords are OK
        connect_key_all = ['host', 'user', 'passwd', 'db', 'charset']
        for connect_key in connect_key_all:
            # if not satisfied
            if not config.__contains__(connect_key):
                print("keyword: '{0}' is needed for connection."
                      .format(connect_key))
                print('Connection failed.')

                err = True

                return err

        # try to make a connection
        try:  # if succeed
            conn = pymysql.connect(host=config['host'], user=config['user'],
                                   passwd=config['passwd'], db=config['db'],
                                   charset=config['charset'])
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

    def article_from_id(self, article_id):
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
            if isinstance(article_id, list):
                articles_str = []
            else:
                articles_str = ''

            err = True

            return articles_str, err

        # get a cursor
        cur = conn.cursor()

        # for only one id
        if not isinstance(article_id, list):

            # announcement
            print('loading an article...')

            # request an article
            cur.execute("select content from mediumContent where id='{0}'".format(article_id))

            # fetch this article
            tmp = cur.fetchall()

            # close the cursor and the connection
            cur.close()
            conn.close()
            print('mySQL server closed.')

            if not tmp == ():  # if exist
                if tmp[0][0] is not None:  # if not empty
                    article_str = tmp[0][0]
                else:
                    article_str = 'NULL'
            else:
                article_str = 'NULL'

            err = False

            return article_str, err

        # for many ids

        # announcement
        print('Begin requesting articles from article ids.')

        # obtain articles via article ID
        articles_str = []

        print('loading articles...')
        time.sleep(0.1)

        # begin a progress bar
        bar = progressbar.ProgressBar()

        # load articles
        for i_d in bar(range(len(article_id))):

            # request an article
            cur.execute("select content from mediumContent where id='{0}'".format(article_id[i_d]))

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

        return articles_str, err

    def recent_articles(self, number=None):
        # load recent articles from mySQL server

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
        request_format = "select id from medium where mediumType = 'article' order by createdAt desc;"
        cur.execute(request_format)

        # fetch data
        data0 = cur.fetchall()

        # end the timer
        time_ed = time.time()

        print('Successfully loaded basic data from the mySQL server. TIME: {0} s'.format(time_ed - time_st))

        # if test mode
        if TEST_MODE:
            data0 = (data0[0],)

        # select recent data
        if number is not None:
            if len(data0) > number:
                data0 = data0[:number]

        print('Total article number: ' + str(len(data0)))

        # get article id
        id_list = list()
        for d in data0:
            id_list.append(d[0])

        # obtain articles via article ID
        articles_str = []

        print('loading articles...')
        time.sleep(0.1)

        # begin a progress bar
        bar = progressbar.ProgressBar()

        # load articles
        for d in bar(id_list):

            # request an article
            cur.execute("select content from mediumContent where id='{0}'".format(d))

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

        return articles_str, id_list, err
