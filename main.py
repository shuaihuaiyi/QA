import readData
import evaluation
import tensorflow as tf

if __name__ == '__main__':
    #定义参数
    trainingFile = "data/training.data"
    testFile = "data/develop.data"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50
