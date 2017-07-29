# -*- coding:utf-8 -*-
import ailab.tasks
from ailab.tasks.multi_tasks import *


class FieldClass(MultiClass):

    __request = 'field'
    __model_dir = ailab.tasks.MODEL_DIR

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(articlesstr, self.__request, self.__model_dir)
        return labelsstr
