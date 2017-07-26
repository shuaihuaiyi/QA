# coding=utf-8

import codecs
import logging
import re
from collections import defaultdict

import jieba
import numpy as np

# define a logger
logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_embedding(filename, embedding_size):
    """
    load embedding
    """
    embeddings = []
    word2idx = defaultdict(list)
    idx2word = defaultdict(list)
    idx = 0
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                idx += 1
                arr = line.split(" ")
                if len(arr) != (embedding_size + 2):
                    logging.error("embedding error, index is:%s" % idx)
                    continue

                embedding = [float(val) for val in arr[1: -1]]
                word2idx[arr[0]] = len(word2idx)
                idx2word[len(word2idx)] = arr[0]
                embeddings.append(embedding)

        except Exception as e:
            logging.error("load embedding Exception,", e)
        finally:
            rf.close()

    logging.info("load embedding finish!")
    return embeddings, word2idx, idx2word


def sent_to_idx(sent, word2idx, sequence_len):
    """
    convert sentence to index array
    """
    unknown_id = word2idx.get("UNKNOWN", 0)
    num_id = word2idx.get("NUM",len(word2idx))
    sent2idx = [unknown_id] * sequence_len
    i = 0
    for word in jieba.cut(sent):
        if(word in word2idx):
            sent2idx[i] = word2idx[word]
        else:
            if re.match("\d+",word):
                sent2idx[i] = num_id
            else:
                sent2idx[i] = unknown_id
        if i >= sequence_len - 1:
            break
        i += 1
    return sent2idx


def loadData(filename, word2idx, sequence_len):
    """
    load data
    """
    ori_quests, cand_quests, labels, questionIds = [], [], [], []
    question = ""
    questionId = 0
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                arr = line.strip().split("\t")
                if len(arr) != 3:
                    logging.error("invalid data:%s" % line)
                    continue
                if question != arr[0]:
                    question = arr[0]
                    questionId += 1
                ori_quest = sent_to_idx(arr[0], word2idx, sequence_len)
                cand_quest = sent_to_idx(arr[1], word2idx, sequence_len)
                label = int(arr[2])

                ori_quests.append(ori_quest)
                cand_quests.append(cand_quest)
                labels.append(label)
                questionIds.append(questionId)

        except Exception as e:
            logging.error("load error,", e)
        finally:
            rf.close()
    logging.info("load data finish!")
    return ori_quests, cand_quests, labels, questionIds


def batch_iter(questions, answers, labels, questionIds, batch_size):
    """
    iterate the data
    """
    trueAnswer = ""
    data_len = questionIds[-1]
    batch_num = int(data_len / batch_size) + 1
    line = 0
    for batch in range(batch_num):
        # 对于每一批问题
        resultQuestions, trueAnswers, falseAnswers = [], [], []
        for questionId in range(batch * batch_size, min((batch + 1) * batch_size, data_len)):
            # 对于每一个问题
            trueCount = 0
            while questionIds[line] == questionId:
                # 对于某个问题中的某一行
                if (labels[line] == 0):
                    resultQuestions.append(questions[line])
                    falseAnswers.append(answers[line])
                else:
                    trueAnswer = answers[line]
                    trueCount += 1
                line += 1
            trueAnswers.extend([trueAnswer] * (questionIds.count(questionId) - trueCount))
        yield np.array(resultQuestions), np.array(trueAnswers), np.array(falseAnswers)


def valid_iter(questions, answers, batch_size):
    lines = len(questions)
    data_len = batch_size * 20
    batch_num = int(lines / data_len) + 1
    questions, answers = np.array(questions), np.array(answers)
    for batch in range(batch_num):
        startIndex = batch * data_len
        endIndex = min(batch * data_len + data_len, lines)
        yield questions[startIndex:endIndex], answers[startIndex:endIndex]
