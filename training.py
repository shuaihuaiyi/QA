import codecs
import tensorflow as tf

#读取训练数据
trainingFile = codecs.open("data/training.data",encoding='utf-8')
lines = trainingFile.readlines()
trainingFile.close()
#解析训练数据
trainingSet = dict()
for line in lines:
    line = line.split('\t')
    line[2] = int(line[2])
    answer = dict([line[1:]])
    if line[0] in trainingSet:
        trainingSet[line[0]].append(answer)
    else:
        trainingSet[line[0]]=[answer]

