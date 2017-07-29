# -*- coding:utf-8 -*-
import ailab.tasks
from ailab.tasks.multi_tasks import *


class ContentTypeClass(MultiClass):

    __request = 'contentType'
    __model_dir = ailab.tasks.MODEL_DIR

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(self, articlesstr, self.__request, self.__model_dir)
        return labelsstr
