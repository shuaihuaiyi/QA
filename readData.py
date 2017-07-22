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
