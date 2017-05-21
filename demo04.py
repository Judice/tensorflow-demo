# coding=utf-8
# cnn 图像识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs: v_xs, keep_pro: 1})
    correct_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_pro: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, 0.1) # 0.1标准误差
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积
def conv2d(x, W):   #变2维      # 步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 中间 1 1 每次横向纵向移1位 区别于valid,same无信息丢失


def max_pool(x):   # 2*2     池化窗2*2
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # mean_pool

# define placeholder for input
xs = tf.placeholder(tf.float32,[None,784])/255  # 归一化
ys = tf.placeholder(tf.float32,[None,10])

#抑制过拟合
keep_pro = tf.placeholder(tf.float32)

x_image= tf.reshape(xs,[-1, 28, 28, 1])   # -1代表样例数 3层变1层

#conv11
W_conv1 = weight_variable([5, 5, 1, 32])  # 提取32个特征  5*5取 输入1幅图
b_conv1= biases_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 激活隐藏层 由于是same 28*28*32
p_con1= max_pool(h_conv1)   # 降维      14*14*32

#con12
W_conv2 = weight_variable([5, 5, 32, 64]) # 提取32个特征  5*5  提出64特征
b_conv2= biases_variable([64])
h_conv2 = tf.nn.relu(conv2d(p_con1, W_conv2)+b_conv2) #激活隐藏层  # 14*14*64
p_con2= max_pool(h_conv2)   #降维  7*7*64 由于是same还是64

# 全连接层
W_fc1 = weight_variable([7*7*64, 1024])  # 变为1维向量
b_fc1= biases_variable([1024])
p_flat = tf.reshape(p_con2, [-1,7*7*64])  # 3维变1维
h_fc1 = tf.nn.relu(tf.matmul(p_flat, W_fc1) + b_fc1) # ?感知机
h_fc1_drop= tf.nn.dropout(h_fc1, keep_pro)  # 避免过拟合 keep_pro为保留数据的比例

#layer2
W_fc2 = weight_variable([1024,10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_x, ys:batch_y,keep_pro:0.5})
    if i % 50 == 0:
        print(computer_accuracy(mnist.test.images,mnist.test.labels))











