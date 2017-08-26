# encoding=utf-8
from __future__ import absolute_import

from ailab.db.db import DB
from simhash import Simhash as sim
from simhash import SimhashIndex
import jieba


class Simhash(object):
    def __init__(self, config):
        self.config = config
        self.db = DB(self.config)
        self.index = SimhashIndex([], k=3)

    def load_recent_simhash(self, num):
        id_list, _, simhash_list, err = self.db.recent_titles_simhash(num)
        if err:
            print 'failed to fetch data'
            return

        for i in range(len(id_list)):
            self.index.add(id_list[i], simhash_list[i])

    @classmethod
    def get_features(cls, s):
        s = list(jieba.cut(s))
        return s

    def calculate(self, article):
        return sim(self.get_features(article))

    def duplicate_or_not(self, article_id, article):
        ss = self.calculate(article)
        self.index.add(article_id, ss)
        return self.index.get_near_dups(ss)
