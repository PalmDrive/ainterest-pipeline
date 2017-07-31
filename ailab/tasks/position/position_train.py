# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import *


class PositionTrain(MultiTrain):

    def __init__(self, output_dir="../../output/"):
        self.__request = 'position'
        self.__output_dir = output_dir

    def train(self):
        labelsstr = MultiTrain.multi_train(self, self.__request, self.__output_dir)
        return labelsstr
