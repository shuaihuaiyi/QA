# coding=utf-8

import codecs
import logging
import numpy as np
import os

from collections import defaultdict

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
                    logging.error("embedding error, index is:%s" % (idx))
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
    sent2idx = [word2idx.get(word, unknown_id) for word in sent.split("_")[:sequence_len]]
    return sent2idx


def load_train_data(filename, word2idx, sequence_len):
    """
    load train data
    """
    ori_quests, cand_quests = [], []
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                arr = line.strip().split(" ")
                if len(arr) != 4 or arr[0] != "1":
                    logging.error("invalid data:%s" % (line))
                    continue

                ori_quest = sent_to_idx(arr[2], word2idx, sequence_len)
                cand_quest = sent_to_idx(arr[3], word2idx, sequence_len)

                ori_quests.append(ori_quest)
                cand_quests.append(cand_quest)

        except Exception as e:
            logging.error("load train data Exception,", e)
        finally:
            rf.close()
    logging.info("load train data finish!")

    return ori_quests, cand_quests


def create_valid(data, proportion=0.1):
    if data is None:
        logging.error("data is none")
        os._exit(1)

    data_len = len(data)
    shuffle_idx = np.random.permutation(np.arange(data_len))
    data = np.array(data)[shuffle_idx]
    seperate_idx = int(data_len * (1 - proportion))
    return data[:seperate_idx], data[seperate_idx:]


def load_test_data(filename, word2idx, sequence_len):
    """
    load test data
    """
    ori_quests, cand_quests, labels, results = [], [], [], []
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        try:
            for line in rf.readlines():
                arr = line.strip().split(" ")
                if len(arr) != 4:
                    logging.error("invalid data:%s" % line)
                    continue

                ori_quest = sent_to_idx(arr[2], word2idx, sequence_len)
                cand_quest = sent_to_idx(arr[3], word2idx, sequence_len)
                label = int(arr[0])
                result = int(arr[1].split(":")[1])

                ori_quests.append(ori_quest)
                cand_quests.append(cand_quest)
                labels.append(label)
                results.append(result)

        except Exception as e:
            logging.error("load test error,", e)
        finally:
            rf.close()
    logging.info("load test data finish!")
    return ori_quests, cand_quests, labels, results


def batch_iter(ori_quests, cand_quests, batch_size, epoches, is_valid=False):
    """
    iterate the data
    """
    data_len = len(ori_quests)
    batch_num = int(data_len / batch_size)
    ori_quests = np.array(ori_quests)
    cand_quests = np.array(cand_quests)

    for epoch in range(epoches):
        if is_valid is not True:
            shuffle_idx = np.random.permutation(np.arange(batch_num * batch_size))
            ori_quests = np.array(ori_quests)[shuffle_idx]
            cand_quests = np.array(cand_quests)[shuffle_idx]
        for batch in range(batch_num):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, data_len)
            act_batch_size = end_idx - start_idx

            # get negative questions
            if is_valid:
                neg_quests = cand_quests[start_idx: end_idx]
            else:
                randi_list = []
                while len(randi_list) != act_batch_size:
                    [randi_list.append(idx) for idx in np.random.randint(0, data_len, 5 * act_batch_size) if
                     start_idx < idx < end_idx and len(randi_list) < act_batch_size]
                neg_quests = [cand_quests[idx] for idx in randi_list]

            yield (ori_quests[start_idx: end_idx], cand_quests[start_idx: end_idx], neg_quests)
