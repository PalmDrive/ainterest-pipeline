# -*- coding:utf-8 -*-
from ailab.similarity.similarity import Similarity


def main():
    # class
    a = Similarity(model_dir='../../../output/similarity')

    # connect
    a.connect('../../config/config.yaml')

    # load recent articles and train a new model
    a.base(20000)

    # save this model
    a.save()

    # clear old models
    a.clear_old_model()

    # load the newest model
    a.load_newest()

    # a new article
    id_str = 'myID'
    article = "PingWest品玩8月21日报道，随着为期两天的（8月19日－20日）首届世界西商大会的闭幕，' \
              '阿里巴巴与西安市政府达成战略合作，前者决定将其西北总部落户西安。根据协议，' \
              '阿里巴巴将在城投、地铁、新零售、智慧城市和大数据方面，与西安市政府进行合作。' \
              '马云表示，西安近年来发展很快，阿里巴巴很期待能参与到大西安的建设中来。' \
              '呈现在协议上，阿里巴巴此次与西安市政府将在以下四个方面进行合作：' \
              '一是蚂蚁金服将在西安加大投资、拓展布局，特别是加快和城投、地铁的合作，' \
              '帮助西安打造科技之都、时尚之都、消费之都；二是以阿里云为平台，参与西安智慧城市建设，' \
              '通过大数据、云计算，促进城市精细化管理，提升城市交通治理水平；' \
              '三是在西安建立阿里巴巴西北总部，发展新零售，同时聚集云计算、金融、电商人才；' \
              '四是和西安的高校、科研院所合作，培养“一带一路”高科技数据人才，打造双创中心。"

    # most similar articles
    sim_article = a.similar(article, id_str, thres=0.67)

    # plot train results
    # if id_str/article is a list, sim_article would be a list, and each element is for one article
    print(sim_article)
    # for d in sim_article:
    #     print(d)


if __name__ == '__main__':
    main()
