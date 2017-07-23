import readData
import evaluation
import tensorflow as tf

if __name__ == '__main__':
    trainingFile = "data/training.data"
    testFile = "data/develop.data"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50
    readData.readEmbeddingFile(embeddingFile,embeddingSize)