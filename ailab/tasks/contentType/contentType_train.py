# -*- coding:utf-8 -*-
import ailab.tasks
from ailab.tasks.multi_tasks import *


class ContentTypeTrain(MultiTrain):

    __request = 'contentType'
    __output_dir = ailab.tasks.MODEL_DIR

    def train(self):
        labelsstr = MultiTrain.multi_train(self, self.__request, self.__output_dir)
        return labelsstr
