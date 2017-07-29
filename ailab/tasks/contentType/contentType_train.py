# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import *


class ContentTypeTrain(MultiTrain):

    __request = 'contentType'
    __output_dir = "../../output/"

    def train(self):
        labelsstr = MultiTrain.multi_train(self, self.__request, self.__output_dir)
        return labelsstr
