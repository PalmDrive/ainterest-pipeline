# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import *


class ContentTypeClass(MultiClass):

    __request = 'contentType'
    __model_dir = "../../output/"

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(articlesstr, self.__request, self.__model_dir)
        return labelsstr
