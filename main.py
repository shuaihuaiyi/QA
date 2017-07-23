import readData
import evaluation
import tensorflow as tf


if __name__ == '__main__':
    # 定义参数
    trainingFile = "data/training.data"
    testFile = "data/develop.data"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50

    # 读取数据
    trainingList = readData.readFile(trainingFile)
    testList = readData.readFile(testFile)
    embeddingDict = readData.readEmbeddingFile(embeddingFile, embeddingSize)

    # 预处理
    trainingVec = readData.textToVec(trainingList, embeddingDict)
    testVec = readData.textToVec(testList, embeddingDict)
