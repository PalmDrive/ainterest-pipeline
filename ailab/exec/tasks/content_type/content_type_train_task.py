# -*- coding:utf-8 -*-
from ailab.tasks.content_type.content_type_train import ContentTypeTrain


def main():
    # class
    a = ContentTypeTrain(output_dir="../../../output/content_type")

    a.connect('../../config/config.yaml')

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
