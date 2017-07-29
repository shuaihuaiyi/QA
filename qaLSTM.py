import tensorflow as tf


class QaLstm(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size):
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_unroll_steps = num_unroll_steps
        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
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
            ori_q = self.biLSTM(ori_quests, self.rnn_size)
            ori_q_feat = tf.nn.tanh(self.max_pooling(ori_q))
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_a = self.biLSTM(cand_quests, self.rnn_size)
            neg_a = self.biLSTM(neg_quests, self.rnn_size)
            cand_q_feat = tf.nn.tanh(self.max_pooling(cand_a))
            neg_q_feat = tf.nn.tanh(self.max_pooling(neg_a))

            test_q_out = self.biLSTM(test_q, self.rnn_size)
            test_q_out = tf.nn.tanh(self.max_pooling(test_q_out))
            test_a_out = self.biLSTM(test_a, self.rnn_size)
            test_a_out = tf.nn.tanh(self.max_pooling(test_a_out))

        self.ori_cand = self.feature2cos_sim(ori_q_feat, cand_q_feat)
        self.ori_neg = self.feature2cos_sim(ori_q_feat, neg_q_feat)
        self.loss, self.acc = self.cal_loss_and_acc(self.ori_cand, self.ori_neg)

        self.test_q_a = self.feature2cos_sim(test_q_out, test_a_out)

    @staticmethod
    def biLSTM(x, hidden_size):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    def feature2cos_sim(feat_q, feat_a):
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
        mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
        cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
        return cos_sim_q_a

    # return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
    @staticmethod
    def max_pooling(lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(
            lstm_out.get_shape()[2])  # (step, length of input for one step)

        # do max-pooling to change the (sequence_length) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        output = tf.reshape(output, [-1, width])

        return output

    @staticmethod
    def cal_loss_and_acc(ori_cand, ori_neg):
        # the target function
        zero = tf.fill(tf.shape(ori_cand), 0.0)
        margin = tf.fill(tf.shape(ori_cand), 0.1)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
            loss = tf.reduce_sum(losses)
            # cal accurancy
        with tf.name_scope("acc"):
            correct = tf.equal(zero, losses)
            acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
        return loss, acc
