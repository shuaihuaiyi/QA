# coding=utf-8

import logging
import time
import tensorflow as tf

import taevaluation

from data_helper import loadData, load_embedding, batch_iter, valid_iter
from polymerization import LstmQa

# ------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "data/training.data", "train corpus file")
tf.flags.DEFINE_string("test_file", "data/develop.data", "test corpus file")
tf.flags.DEFINE_string("valid_file", "data/develop.data", "test corpus file")
tf.flags.DEFINE_string("result_file", "predictRst.score", "result file")
tf.flags.DEFINE_string("embedding_file", "word2vec\zhwiki_2017_03.sg_50d.word2vec", "embedding file")
tf.flags.DEFINE_integer("embedding_size", 50, "embedding size")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
learningRate = 0.5
tf.flags.DEFINE_integer("batch_size", 20, "batch size of each batch")
tf.flags.DEFINE_integer("epochs", 20, "epochs")
tf.flags.DEFINE_integer("rnn_size", 100, "rnn size")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "embedding size")
tf.flags.DEFINE_integer("num_unroll_steps", 100, "句子中的最大词汇数目")
# tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_float("gpu_options", 0.75, "use memory rate")

FLAGS = tf.flags.FLAGS

logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)

embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
train_questions, train_answers, train_labels, train_questionId = loadData(FLAGS.train_file, word2idx,
                                                                          FLAGS.num_unroll_steps)

test_questions, test_answers, test_labels, test_questionId = loadData(FLAGS.test_file, word2idx, FLAGS.num_unroll_steps)
valid_questions, valid_answers, valid_labels, valid_questionId = loadData(FLAGS.valid_file, word2idx,
                                                                          FLAGS.num_unroll_steps)


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
    # time_str = datetime.datetime.now().isoformat()
    # right, wrong, score = [0.0] * 3
    # for i in range(0, len(ori_batch)):
    #     if ori_cand_score[i] > 0.55 and ori_neg_score[i] < 0.4:
    #         right += 1.0
    #     else:
    #         wrong += 1.0
    #     score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    print("step=",step," time=",time_elapsed)
    # logger.info("%s: step %s, loss %s, acc %s, score %s, wrong %s, %6.7f secs/batch" % (
    #     time_str, step, cur_loss, cur_acc, score, wrong, time_elapsed))

    return cur_loss, ori_cand_score


def valid_run_step(sess, ori_batch, cand_batch, lstm, dropout=1.):
    feed_dict = {
        lstm.test_input_q: ori_batch,
        lstm.test_input_a: cand_batch,
        lstm.keep_prob: dropout
    }

    step, ori_cand_score = sess.run([global_step, lstm.test_q_a], feed_dict)

    return ori_cand_score


def valid_model(sess, lstm, valid_questions, valid_answers, valid_file, result_file, valid = True):
    # 输出文件
    with open(result_file, 'w') as file:
        for questions, answers in valid_iter(valid_questions, valid_answers, FLAGS.batch_size):
            scores = valid_run_step(sess, questions, answers, lstm)
            for score in scores:
                file.write("%.9f" % score + '\n')
    if valid:
        # 评估MRR
        file.close()
        taevaluation.evaluate(valid_file, result_file)


with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            lstm = LstmQa(FLAGS.batch_size, FLAGS.num_unroll_steps, embedding, FLAGS.embedding_size, FLAGS.rnn_size,
                          FLAGS.num_rnn_layers)
            global_step = tf.Variable(0, name="globle_step", trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                              FLAGS.max_grad_norm)

            sess.run(tf.global_variables_initializer())
            tqs,tta,tfa = [],[],[]
            for ori_train, cand_train, neg_train in batch_iter(train_questions, train_answers,
                                                               train_labels, train_questionId, FLAGS.batch_size):
                tqs.append(ori_train),tta.append(cand_train),tfa.append(neg_train)
            for i in range(3):
                train_op = tf.train.GradientDescentOptimizer(learningRate).apply_gradients(zip(grads, tvars),global_step=global_step)
                for epoch in range(FLAGS.epochs):
                    for ori_train, cand_train, neg_train in zip(tqs,tta,tfa):
                        run_step(sess, ori_train, cand_train, neg_train, lstm)
                    valid_model(sess, lstm, valid_questions, valid_answers, FLAGS.valid_file, FLAGS.result_file)
                learningRate/=2
            valid_model(sess, lstm, test_questions, test_answers, FLAGS.test_file, FLAGS.result_file,False)
