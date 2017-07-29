import tensorflow as tf


class QaLSTMNet(object):
    """
    中文问答系统使用的LSTM网络结构
    """

    def __init__(self, batchSize, unrollSteps, embeddings, embeddingSize, rnnSize, margin):
        self.batchSize = batchSize
        self.unrollSteps = unrollSteps
        self.embeddings = embeddings
        self.embeddingSize = embeddingSize
        self.rnnSize = rnnSize
        self.margin = margin

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])

        # 设置word embedding层
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            tfEmbedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            questions = tf.nn.embedding_lookup(tfEmbedding, self.inputQuestions)
            trueAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTrueAnswers)
            falseAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputFalseAnswers)

            testQuestions = tf.nn.embedding_lookup(tfEmbedding, self.inputTestQuestions)
            testAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTestAnswers)

        # 建立LSTM网络
        with tf.variable_scope("LSTM_scope", reuse=None):
            question1 = self.biLSTMCell(questions, self.rnnSize)
            question2 = tf.nn.tanh(self.max_pooling(question1))
        with tf.variable_scope("LSTM_scope", reuse=True):
            trueAnswer1 = self.biLSTMCell(trueAnswers, self.rnnSize)
            trueAnswer2 = tf.nn.tanh(self.max_pooling(trueAnswer1))
            falseAnswer1 = self.biLSTMCell(falseAnswers, self.rnnSize)
            falseAnswer2 = tf.nn.tanh(self.max_pooling(falseAnswer1))

            testQuestion1 = self.biLSTMCell(testQuestions, self.rnnSize)
            testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
            testAnswer1 = self.biLSTMCell(testAnswers, self.rnnSize)
            testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))

        self.trueCosSim = self.getCosineSimilarity(question2, trueAnswer2)
        self.falseCosSim = self.getCosineSimilarity(question2, falseAnswer2)
        self.loss = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)

        self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)

    @staticmethod
    def biLSTMCell(x, hiddenSize):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    def getCosineSimilarity(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def getLoss(trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss
