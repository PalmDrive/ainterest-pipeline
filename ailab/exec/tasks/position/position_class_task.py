# -*- coding:utf-8 -*-
from ailab.tasks.position.position_class import PositionClass


def main():
    # data
    # articles_str = get_data_file()
    articles_str_1 = 'This is a good classifier.'
    articles_str_2 = ['This is a good classifier.', '也可以处理列表']

    # class
    a = PositionClass()

    a.algorithm('l1dcd')

    # classify
    b1 = a.classify(articles_str_1)
    b2 = a.classify(articles_str_2)

    a.algorithm('libsvm')

    # classify
    b3 = a.classify(articles_str_1)
    b4 = a.classify(articles_str_2)

    print("\n")
    print('l1dcd:')
    print(b1)
    print(b2)
    print("\n")

    print("\n")
    print('LIBSVM:')
    print(b3)
    print(b4)
    print("\n")


if __name__ == '__main__':
    main()
