# coding=utf-8

import logging
import datetime
import time
import tensorflow as tf
import operator

from data_helper import load_train_data, load_test_data, load_embedding, batch_iter
from polymerization import LSTM_QA

# ------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../insuranceQA/train", "train corpus file")
tf.flags.DEFINE_string("test_file", "../insuranceQA/test1", "test corpus file")
tf.flags.DEFINE_string("valid_file", "../insuranceQA/test1.sample", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../insuranceQA/vectors.nobin", "embedding file")
tf.flags.DEFINE_integer("embedding_size", 100, "embedding size")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
tf.flags.DEFINE_float("lr", 0.1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 100, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 300, "epoches")
tf.flags.DEFINE_integer("rnn_size", 300, "embedding size")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "embedding size")
tf.flags.DEFINE_integer("evaluate_every", 1000, "run evaluation")
tf.flags.DEFINE_integer("num_unroll_steps", 100, "embedding size")
tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.75, "use memory rate")

FLAGS = tf.flags.FLAGS
# ----------------------------- define parameter end ----------------------------------

# ----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)
# ----------------------------- define a logger end ----------------------------------

# ------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
ori_quests, cand_quests = load_train_data(FLAGS.train_file, word2idx, FLAGS.num_unroll_steps)

test_ori_quests, test_cand_quests, labels, results = load_test_data(FLAGS.test_file, word2idx, FLAGS.num_unroll_steps)
valid_ori_quests, valid_cand_quests, valid_labels, valid_results = load_test_data(FLAGS.valid_file, word2idx,
                                                                                  FLAGS.num_unroll_steps)


# ----------------------------------- load data end ----------------------

# ----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch, lstm, dropout=1.):
    start_time = time.time()
    feed_dict = {
        lstm.ori_input_quests: ori_batch,
        lstm.cand_input_quests: cand_batch,
        lstm.neg_input_quests: neg_batch,
        lstm.keep_prob: dropout
    }

    _, step, ori_cand_score, ori_neg_score, cur_loss, cur_acc = sess.run(
        [train_op, global_step, lstm.ori_cand, lstm.ori_neg, lstm.loss, lstm.acc], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    right, wrong, score = [0.0] * 3
    for i in range(0, len(ori_batch)):
        if ori_cand_score[i] > 0.55 and ori_neg_score[i] < 0.4:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    logger.info("%s: step %s, loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch" % (
    time_str, step, cur_loss, cur_acc, score, wrong, time_elapsed))

    return cur_loss, ori_cand_score


def valid_run_step(sess, ori_batch, cand_batch, lstm, dropout=1.):
    feed_dict = {
        lstm.test_input_q: ori_batch,
        lstm.test_input_a: cand_batch,
        lstm.keep_prob: dropout
    }

    step, ori_cand_score = sess.run([global_step, lstm.test_q_a], feed_dict)

    return ori_cand_score


# ---------------------------------- execute train model end --------------------------------------

def cal_acc(labels, results, total_ori_cand):
    if len(labels) == len(results) == len(total_ori_cand):
        retdict = {}
        for label, result, ori_cand in zip(labels, results, total_ori_cand):
            if result not in retdict:
                retdict[result] = []
            retdict[result].append((ori_cand, label))

        correct = 0
        for key, value in retdict.items():
            value.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = value[0]
            if flag == 1:
                correct += 1
        return 1. * correct / len(retdict)
    else:
        logger.info("data error")
        return 0


# ---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, lstm, valid_ori_quests, valid_cand_quests, labels, results):
    logger.info("start to validate model")
    total_ori_cand = []
    for ori_valid, cand_valid, neg_valid in batch_iter(valid_ori_quests, valid_cand_quests, FLAGS.batch_size, 1,
                                                       is_valid=True):
        ori_cand = valid_run_step(sess, ori_valid, cand_valid, lstm)
        total_ori_cand.extend(ori_cand)

    data_len = len(total_ori_cand)
    acc = cal_acc(labels[:data_len], results[:data_len], total_ori_cand)
    timestr = datetime.datetime.now().isoformat()
    logger.info("%s, evaluation acc:%s" % (timestr, acc))


# ---------------------------------- execute valid model end --------------------------------------

# ----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement,
                                      gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            lstm = LSTM_QA(FLAGS.batch_size, FLAGS.num_unroll_steps, embedding, FLAGS.embedding_size, FLAGS.rnn_size,
                           FLAGS.num_rnn_layers, FLAGS.max_grad_norm)
            global_step = tf.Variable(0, name="globle_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                              FLAGS.max_grad_norm)

            # optimizer = tf.train.GradientDescentOptimizer(lstm.lr)
            optimizer = tf.train.GradientDescentOptimizer(1e-1)
            optimizer.apply_gradients(zip(grads, tvars))
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.epoches):
                # cur_lr = FLAGS.lr / (epoch + 1)
                # lstm.assign_new_lr(sess, cur_lr)
                # logger.info("current learning ratio:" + str(cur_lr))
                for ori_train, cand_train, neg_train in batch_iter(ori_quests, cand_quests, FLAGS.batch_size,
                                                                   epoches=1):
                    run_step(sess, ori_train, cand_train, neg_train, lstm)
                    cur_step = tf.train.global_step(sess, global_step)

                    if cur_step % FLAGS.evaluate_every == 0 and cur_step != 0:
                        valid_model(sess, lstm, valid_ori_quests, valid_cand_quests, valid_labels, valid_results)
            valid_model(sess, lstm, test_ori_quests, test_cand_quests, labels, results)
            # ---------------------------------- end train -----------------------------------
