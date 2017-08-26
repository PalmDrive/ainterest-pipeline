# -*- coding:utf-8 -*-
from ailab.similarity.simhash import Simhash


def main():
    # class
    a = Simhash(config='../../config/config.yaml')
    #a.load_recent_simhash(5)
    # a new article
    id_str = 'myID'
    article = u"wo我节日哦安乐的师傅看见啊手机费 i 哦挖掘饿啊李开复卡勒季斯的风景哦爱就违法礼卡就是快乐的卷发哦 is 的减肥了啊我看见俄方哦爱就啥都发生的快放假啦说的减肥啊哦 is 的家发了说的减肥克拉结束的立法精神动力飞机啊老师都快解放啦是江东父老啊是江东父老啊绝世独立封口机阿拉山口的减肥阿里看到减肥la"

    # most similar articles
    sim_article = a.duplicate_or_not(id_str, article)

    # plot train results
    # if id_str/article is a list, sim_article would be a list, and each element is for one article
    print(sim_article)
    # for d in sim_article:
    #     print(d)
    nn = u"wo我节日哦安乐的师傅看见啊手机费 i,!@#$%^&*()7891902312,.?: 哦挖掘饿啊李开复卡勒季斯的风景哦爱就违法礼卡就是快乐的卷发哦 is 的减肥了啊我看见俄方哦爱就啥都发生的快放假啦说的减肥啊哦 is 的家发了说的减肥克拉结束的立法精神动力飞机啊老师都快解放啦是江东父老啊是江东父老啊绝世独立封口机阿拉山口的减肥阿里看到减肥la"

    sim_article = a.duplicate_or_not("nllk", article)

    # plot train results
    # if id_str/article is a list, sim_article would be a list, and each element is for one article
    print(sim_article)
    # for d in sim_article:
    #     print(d)


if __name__ == '__main__':
    main()
