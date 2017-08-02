# -*- coding:utf-8 -*-
from tasks.position.position_train import PositionTrain


def main():
    # class
    a = PositionTrain()

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
