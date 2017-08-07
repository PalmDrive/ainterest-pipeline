# -*- coding:utf-8 -*-
from ailab.tasks.content_type.content_type_train import ContentTypeTrain


def main():
    # class
    a = ContentTypeTrain(output_dir="../../../output/content_type")

    a.connect()

    # algorithm
    a.algorithm('libsvm')

    a.param = {'libsvm': '-t 2 -c 10 -g 0.01 -e 0.00001'}

    # thread
    a.thread(8)

    # train
    a.train()

    # save model
    a.save()

    # plot train results
    a.plot()


if __name__ == '__main__':
    main()
