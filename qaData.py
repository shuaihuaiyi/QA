import re
from collections import defaultdict

import jieba
import numpy as np


def loadEmbedding(filename):
    """
    加载词向量文件

    :param filename: 文件名
    :return: embeddings列表和它对应的索引
    """
    embeddings = []
    word2idx = defaultdict(list)
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf:
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)
    return embeddings, word2idx


def sentenceToIndex(sentence, word2idx, maxLen):
    """
    将句子分词，并转换成embeddings列表的索引值

    :param sentence: 句子
    :param word2idx: 词语的索引
    :param maxLen: 句子的最大长度
    :return: 句子的词向量索引表示
    """
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * maxLen
    i = 0
    for word in jieba.cut(sentence):
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= maxLen - 1:
            break
        i += 1
    return index


def loadData(filename, word2idx, maxLen, training=False):
    """
    加载训练文件或者测试文件

    :param filename: 文件名
    :param word2idx: 词向量索引
    :param maxLen: 句子的最大长度
    :param training: 是否作为训练文件读取
    :return: 问题，答案，标签和问题ID
    """
    question = ""
    questionId = 0
    questions, answers, labels, questionIds = [], [], [], []
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf.readlines():
            arr = line.split("\t")
            if question != arr[0]:
                question = arr[0]
                questionId += 1
            questionIdx = sentenceToIndex(arr[0].strip(), word2idx, maxLen)
            answerIdx = sentenceToIndex(arr[1].strip(), word2idx, maxLen)
            if training:
                label = int(arr[2])
                labels.append(label)
            questions.append(questionIdx)
            answers.append(answerIdx)
            questionIds.append(questionId)
    return questions, answers, labels, questionIds


def trainingBatchIter(questions, answers, labels, questionIds, batchSize):
    """
    逐个获取每一批训练数据的迭代器，会区分每个问题的正确和错误答案，拼接为（q，a+，a-）形式

    :param questions: 问题列表
    :param answers: 答案列表
    :param labels: 标签列表
    :param questionIds: 问题ID列表
    :param batchSize: 每个batch的大小
    """
    trueAnswer = ""
    dataLen = questionIds[-1]
    batchNum = int(dataLen / batchSize) + 1
    line = 0
    for batch in range(batchNum):
        # 对于每一批问题
        resultQuestions, trueAnswers, falseAnswers = [], [], []
        for questionId in range(batch * batchSize, min((batch + 1) * batchSize, dataLen)):
            # 对于每一个问题
            trueCount = 0
            while questionIds[line] == questionId:
                # 对于某个问题中的某一行
                if labels[line] == 0:
                    resultQuestions.append(questions[line])
                    falseAnswers.append(answers[line])
                else:
                    trueAnswer = answers[line]
                    trueCount += 1
                line += 1
            trueAnswers.extend([trueAnswer] * (questionIds.count(questionId) - trueCount))
        yield np.array(resultQuestions), np.array(trueAnswers), np.array(falseAnswers)


def testingBatchIter(questions, answers, batchSize):
    """
    逐个获取每一批测试数据的迭代器

    :param questions: 问题列表
    :param answers: 答案列表
    :param batchSize: 每个batch的大小
    """
    lines = len(questions)
    dataLen = batchSize * 20
    batchNum = int(lines / dataLen) + 1
    questions, answers = np.array(questions), np.array(answers)
    for batch in range(batchNum):
        startIndex = batch * dataLen
        endIndex = min(batch * dataLen + dataLen, lines)
        yield questions[startIndex:endIndex], answers[startIndex:endIndex]
