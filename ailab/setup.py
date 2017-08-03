from setuptools import setup
from setuptools import find_packages


setup(name='ailab',
      version='0.0.1',
      description='ailab for ai gateway',
      install_requires=['numpy', 'progressbar', 'pymysql',
                        'jieba', ],
      extras_require={},
      packages=find_packages())
