import tensorflow as tf


# # ----------------------------- cal attention -------------------------------
# # input_q, input_a (batch_size, rnn_size, seq_len)
# def cal_attention(input_q, input_a, U):
#     batch_size = int(input_q.get_shape()[0])
#     U = tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])
#     G = tf.matmul(tf.matmul(input_q, U, True), input_a)
#     delta_q = tf.nn.softmax(tf.reduce_max(G, 1), 1)
#     delta_a = tf.nn.softmax(tf.reduce_max(G, 2), 1)
#
#     return delta_q, delta_a


def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return cos_sim_q_a


# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_length) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])

    return output


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
