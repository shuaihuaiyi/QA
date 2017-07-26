import tensorflow as tf
import readData
import operator


def readResult(resultFileName):
    """
    读取并解析输出文件中的数据

    :param resultFileName: 输出文件的路径
    :return: 数据集合
    """
    with open(resultFileName, encoding='utf-8') as inputFile:
        result = []
        for line in inputFile:
            line = float(line)
            result.append(line)
    return result


def getMRR(testFile, resultFile):
    """
    求本模型的MRR得分。若第一个正确答案排在第n位，则MRR得分就是1/n

    :param testFile: 测试集的文件路径
    :param resultFile: 输出结果的文件路径
    :return: MRR得分值
    """
    testSet = readData.readFile(testFile)
    resultSet = readResult(resultFile)
    currentQuestion = testSet[0][0]
    questionMap = []
    sumq = 0.0
    correct = 0.0
    for i in range(len(testSet)):
        line = testSet[i]
        if line[0] != currentQuestion:
            currentQuestion = line[0]
            questionMap.sort(key=operator.itemgetter(0))
            sumq+=1.0
            for j in range(len(questionMap)):
                if questionMap[j][1] == 1.0:
                    correct+=1/(j+1)
                    break
            questionMap.clear()
        questionMap.append([resultSet[i],line[2]])
    sumq += 1.0
    for j in range(len(questionMap)):
        if questionMap[j][1] == 1.0:
            correct += 1 / (j + 1)
    return correct/sumq
