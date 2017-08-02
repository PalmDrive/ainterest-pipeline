# -*- coding:utf-8 -*-
from tasks.multi_tasks import MultiClass


class FieldClass(MultiClass):

    def __init__(self, model_dir="../../output/"):
        MultiClass.__init__(self)

        # not editable
        self.__request = 'field'
        self.__model_dir = model_dir

    def classify(self, articlesstr):
        labelsstr = MultiClass.multi_classify(self, articlesstr, self.__request, self.__model_dir)
        return labelsstr
