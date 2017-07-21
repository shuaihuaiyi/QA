import codecs
import tensorflow as tf

# 读取并解析训练数据
with codecs.open("data/training.data", encoding='utf-8') as trainingFile:
    trainingSet = dict()
    for line in trainingFile:
        line = line.split('\t')
        line[2] = float(line[2])
        answer = dict([line[1:]])
        if line[0] in trainingSet:
            trainingSet[line[0]].append(answer)
        else:
            trainingSet[line[0]] = [answer]
