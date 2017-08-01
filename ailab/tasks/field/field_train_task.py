# -*- coding:utf-8 -*-
from ailab.tasks.field.field_train import FieldTrain


def main():
    # class
    a = FieldTrain()

    a.connect({'host': '127.0.0.1', 'user': 'user',
               'passwd': 'passwd', 'db': 'db'})

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
