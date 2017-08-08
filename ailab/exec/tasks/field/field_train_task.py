# -*- coding:utf-8 -*-
from ailab.tasks.field.field_train import FieldTrain


def main():
    # class
    a = FieldTrain(output_dir="../../../output/field")

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
