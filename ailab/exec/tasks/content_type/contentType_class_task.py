# -*- coding:utf-8 -*-
from ailab.tasks.contentType.contentType_class import ContentTypeClass


def main():
    # data
    # articlesstr = get_data_file()
    articlesstr_1 = 'This is a good classifier.'
    articlesstr_2 = ['This is a good classifier.', '也可以处理列表']

    # class
    a = ContentTypeClass()

    a.algorithm('l1dcd')

    # classify
    b1 = a.classify(articlesstr_1)
    b2 = a.classify(articlesstr_2)

    a.algorithm('libsvm')

    # classify
    b3 = a.classify(articlesstr_1)
    b4 = a.classify(articlesstr_2)

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
