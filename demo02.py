import tensorflow as tf
import numpy as np
# 线性函数回归
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.7 + 0.5

weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros(1))
y = x_data*weight + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(sess.run(weight),sess.run(biases))




