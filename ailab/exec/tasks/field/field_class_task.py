# -*- coding:utf-8 -*-
from ailab.tasks.field.field_class import FieldClass


def main():
    # data
    # articles_str = get_data_file()
    articles_str_1 = '心理学, 诸如, 各位, 经典, 艺术'
    articles_str_2 = ['This is a good classifier.', '也可以处理列表']

    # class
    a = FieldClass()

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
