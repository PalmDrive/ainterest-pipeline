# -*- coding:utf-8 -*-
from ailab.tasks.position.position_train import PositionTrain


def main():
    # class
    a = PositionTrain(output_dir="../../../output/position")

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
