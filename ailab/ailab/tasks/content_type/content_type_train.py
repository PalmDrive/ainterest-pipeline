# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import MultiTrain


class ContentTypeTrain(MultiTrain):

    def __init__(self, output_dir="../../../output/contentType"):
        MultiTrain.__init__(self)

        # not editable
        self.__request = 'contentType'
        self.__output_dir = output_dir

    def train(self):
        MultiTrain.multi_train(self, self.__request, self.__output_dir)

    def save(self):
        MultiTrain.multi_save(self, self.__output_dir)
