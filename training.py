import codecs
import tensorflow as tf

#读取训练数据
trainingFile = codecs.open("data/training.data",encoding='utf-8')
trainingSet = trainingFile.readlines()
trainingFile.close()
