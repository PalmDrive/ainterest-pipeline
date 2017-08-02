# -*- coding:utf-8 -*-
from tasks.contentType.contentType_train import ContentTypeTrain


def main():
    # class
    a = ContentTypeTrain()

    a.connect({'host': '127.0.0.1', 'user': 'myaccount', 'passwd': 'mypassword',
               'db': 'database', 'charset': 'utf8'})

    # algorithm
    a.algorithm('l1dcd')

    # thread
    a.thread(4)

    # train
    a.train()

    # save model
    a.save()

    # plot train results
    a.plot()


if __name__ == '__main__':
    main()
