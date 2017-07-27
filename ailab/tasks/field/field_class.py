# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import *


class FieldClass(MultiClass):

    __request = 'field'
    __model_dir = "../../output/"

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(articlesstr, self.__request, self.__model_dir)
        return labelsstr
