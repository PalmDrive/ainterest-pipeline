- [x] Requirements file
- [x] stop_words.txt
- [x] Char-level Convolutional neural network model building and some basic analysing graph: **charCNN.py **
- [x] Word-embedding Convolutional neural network model building and some basic analysing graph: **wordEmCNN.py**
- [x] scripts for utils include batch loader, chinese2character converter etc. : **utils.py**
- [x] Script for annotation: **annotate.py**
- [x] **cnn_runner.ipynb** can be executed on jupyter notebook

Project Description:

这个分类器使用的是本地数据，所以没有加入连接数据库的部分，但是由于存在完整的pipeline， 应该可以很容易可以加入数据库的部分。



本分类器的目的是给文章进行分类标注。==输入数据==是文章和其预先标注的pair 集。==训练结果==是一个可保存的数学模型pipeline，可以实现输入文章并输出标注的label的index， 而后通过index-label mapping 可以得到模型预测出的文本标注（由于测试过程中不需要这个功能所以mapping过程未实现）。预处理部分使用了jieba和pypinyin进行中文分词和中文到拼音（alphabets，仅在character level cnn中用到。使用了[fastText](https://research.fb.com/fasttext/) 提供的方法从文本集合获得了word embedding，分类器核心：

1. word embedding based cnn（事先将中文预料map到向量空间中，使用==向量==来表示语义关系和词性等语料特性，转化后的一篇文章便成为了一个由向量组成的矩阵。将矩阵作为一个sample输入卷积网络。

2. character level based cnn (将中文文本分词后转化为alphabets，map使用“dict1” ："abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n" 的字典，转化过程例如：a -> [1 0 0 0 0 0 …] (with a length of “dict1”). 之后的步骤同上。

   ​

   * Convolutional neural network 使用在训练的过程中，会不断调整一层kernal的参数，使得kernal生成的feature map符合模型上层 信息的反馈（通过backpropgation给予）这是一个不断在各层layer之间正向迭代与反向传播的过程。最终目标是尽可能降低预测结果与真实值的差距即loss。

输入数据格式：python list: [文章1,文章2, ……] 

