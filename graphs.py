import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 5, 5, 3])
y = tf.Variable(tf.zeros([5, 5, 3, 16]))
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b
print y