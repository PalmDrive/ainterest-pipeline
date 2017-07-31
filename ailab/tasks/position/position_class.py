# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import *


class PositionClass(MultiClass):

    def __init__(self, model_dir="../../output/"):
        self.__request = 'position'
        self.__model_dir = model_dir

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(self, articlesstr, self.__request, self.__model_dir)
        return labelsstr
