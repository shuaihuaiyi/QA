# coding=utf-8

import time
import tensorflow as tf

import taevaluation

from qaData import loadData, loadEmbedding, batchIter, valid_iter
from qaLSTM import QaLstm





def valid_run_step(sess, ori_batch, cand_batch, lstm, dropout=1.):
    feed_dict = {
        lstm.test_input_q: ori_batch,
        lstm.test_input_a: cand_batch,
        lstm.keep_prob: dropout
    }

    step, ori_cand_score = sess.run([global_step, lstm.test_q_a], feed_dict)

    return ori_cand_score


def valid_model(sess, lstm, valid_questions, valid_answers, valid_file, result_file, valid=True):
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













            saver.restore(sess,'models/79'+saveFile)
            valid_model(sess, lstm, valid_questions, valid_answers, FLAGS.valid_file, FLAGS.result_file)
            valid_model(sess, lstm, test_questions, test_answers, FLAGS.test_file, FLAGS.result_file,False)
