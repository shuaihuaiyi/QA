# coding:utf-8
import tensorflow as tf
from bilstm import biLSTM
from utils import feature2cos_sim, max_pooling, cal_loss_and_acc


class LstmQa(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size, num_rnn_layers,
                 max_grad_norm, l2_reg_lambda=0.0, adjust_weight=False, label_weight=None, is_training=True):
        # define input variable
        if label_weight is None:
            label_weight = []
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_unroll_steps = num_unroll_steps
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")

        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.neg_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])

        self.test_input_q = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.test_input_a = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])

        # embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            ori_quests = tf.nn.embedding_lookup(W, self.ori_input_quests)
            cand_quests = tf.nn.embedding_lookup(W, self.cand_input_quests)
            neg_quests = tf.nn.embedding_lookup(W, self.neg_input_quests)

            test_q = tf.nn.embedding_lookup(W, self.test_input_q)
            test_a = tf.nn.embedding_lookup(W, self.test_input_a)

        # build LSTM network
        with tf.variable_scope("LSTM_scope", reuse=None):
            ori_q = biLSTM(ori_quests, self.rnn_size)
            ori_q_feat = tf.nn.tanh(max_pooling(ori_q))
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_a = biLSTM(cand_quests, self.rnn_size)
            neg_a = biLSTM(neg_quests, self.rnn_size)
            cand_q_feat = tf.nn.tanh(max_pooling(cand_a))
            neg_q_feat = tf.nn.tanh(max_pooling(neg_a))

            test_q_out = biLSTM(test_q, self.rnn_size)
            test_q_out = tf.nn.tanh(max_pooling(test_q_out))
            test_a_out = biLSTM(test_a, self.rnn_size)
            test_a_out = tf.nn.tanh(max_pooling(test_a_out))

        self.ori_cand = feature2cos_sim(ori_q_feat, cand_q_feat)
        self.ori_neg = feature2cos_sim(ori_q_feat, neg_q_feat)
        self.loss, self.acc = cal_loss_and_acc(self.ori_cand, self.ori_neg)

        self.test_q_a = feature2cos_sim(test_q_out, test_a_out)

    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr: lr_value})
