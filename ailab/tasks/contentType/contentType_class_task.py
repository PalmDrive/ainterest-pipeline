# -*- coding:utf-8 -*-
from ailab.tasks.contentType.contentType_class import *


def main():
    # Get data
    # articlesstr = get_data_file()
    articlesstr_1 = 'This is a good classifier.'
    articlesstr_2 = ['This is a good classifier.', '也可以处理列表']

    # class
    aclass = ContentTypeClass()

    # classify
    aclass.classify(articlesstr_1)
    aclass.classify(articlesstr_2)


if __name__ == '__main__':
    main()
