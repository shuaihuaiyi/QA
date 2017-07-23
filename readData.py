import tensorflow as tf


def readFile(filename):
    """
    读取并解析输入文件（训练集和测试集）中的数据

    :param filename: 到指定文件的路径字符串
    :return: 矩阵表示的文件内容
    """
    with open(filename, encoding='utf-8') as inputFile:
        result = []
        for line in inputFile:
            line = line.split('\t')
            line[2] = float(line[2])
            result.append(line)
    return result


def readEmbeddingFile(filename, embeddingSize):
    """
    读取训练好的word2vec文件，获取每个词语对应的向量

    :param filename:文件名
    :param embeddingSize:转换成的词向量的维度
    :return:词语到词向量的映射字典，注意这个词典中未登录词的处理
    """
    word2idx = dict()
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf.readlines():
            arr = line.split(" ")
            if len(arr) != (embeddingSize + 2):
                print("词向量维度定义有误")
                continue
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = embedding
    return word2idx
