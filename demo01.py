import tensorflow as tf
import numpy.py as np
# neural network
def add_layer(input, in_size, out_size, activation_functon=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))  # 行 列  正太分布
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 1行 outsize列
    z = tf.matmul(input, Weight) + biases  # z=wx+b
    if activation_functon is None:
        outputs = z
    else:
        outputs = activation_functon(z)
    return outputs

# make up some data
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]  # 增加维数 -1到1间均等取300个数,生成300*1维矩阵
noise = np.random.normal(0, 0.05, x_data.shape)  #
y_data = np.square(x_data)-0.5+noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,1])   # n*1维占位符 即300*1
ys = tf.placeholder(tf.float32, [None,1])   # 也为300*1矩阵

# hidden layer

l1 = add_layer(xs, 1, 10, activation_functon=tf.nn.relu) # 输入到隐藏,生成300*10维矩阵

# add output layer
prediction = add_layer(l1, 10, 1, activation_functon=None) # 隐藏到输出, 生成300*1维矩阵
# 求均方差
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))  # [1]为行求和 [0]为列求和,矩阵300行数值平方后,每行求和,即不变

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for step in range(1000):
    sess.run(train_step,feed_dict={xs:x_data, ys:y_data})
    if step%50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
