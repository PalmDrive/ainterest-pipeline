from simhash import Simhash as sim
from simhash import SimhashIndex
import jieba


class Simhash(object):

    def __init__(self, objs):
        self.index = SimhashIndex(objs, k=3)

    @classmethod
    def get_features(cls, s):
        width = 3
        s = jieba.cut(s)
        return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

    def calculate(self, article):
        return sim(self.get_features(article)).value

    def duplicate_or_not(self, article_id, article):
        s = self.calculate(article)
        self.index.add(article_id, s)
        return self.index.get_near_dups(s)
