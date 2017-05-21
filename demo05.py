# coding=utf-8
# RNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.set_random_seed(1)
lr = 0.001
train_iters = 1000
n_hidden_units = 128
n_class = 10    # 10个label
n_input = 28    # 1行28列
n_step = 28   # 28行
batch_size = 128

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None,n_class])

weight = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}


def RNN(X, weight, biases):
    # [128,28,128] [128*28,128]
    X = tf.reshape(X, [-1, n_input])
    X_in = tf.matmul(X, weight['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])  # 转换为2维
    #cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias = 1.0,state_is_tupe=True)

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weight['out']) + biases['out']
    return results

pre = RNN(x, weight, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)     # learning rate
correct_pre = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    step = 0
    sess.run(init)
    while step < train_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_step, n_input])
        sess.run([train_op],feed_dict={x: batch_x, y: batch_y})
        step += 1
        if step % 20 == 0 :
            print(sess.run(acc,feed_dict={x: batch_x, y: batch_y}))



