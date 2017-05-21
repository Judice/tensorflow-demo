# coding=utf-8
# neural network 图像识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(input, in_size, out_size, activation_functon=None):
    weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    z = tf.matmul(input,weight) + biases
    if activation_functon == None:
        output = z
    else:
        output = activation_functon(z)
    return output

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholer for input
xs = tf.placeholder(tf.float32, [None,784])  # 28*28
ys = tf.placeholder(tf.float32, [None,10])

# output
prediction = add_layer(xs,784,10,activation_functon=tf.nn.softmax) # 交叉熵

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

