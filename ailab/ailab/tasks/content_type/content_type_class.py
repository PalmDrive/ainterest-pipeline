# -*- coding:utf-8 -*-
from ailab.tasks.multi_tasks import MultiClass


class ContentTypeClass(MultiClass):

    def __init__(self, model_dir="../../../output/content_type"):
        MultiClass.__init__(self)

        # not editable
        self.__request = 'content_type'
        self.__model_dir = model_dir

    def classify(self, articles_str):
        labels_str = MultiClass.multi_classify(self, articles_str, self.__model_dir)
        return labels_str
