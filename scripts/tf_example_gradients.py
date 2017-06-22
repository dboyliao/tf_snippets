#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import tensorflow as tf
import os
# shut up Cpp logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' 

def grad_np(Y, X, beta):
    """
    square error gradient: numpy implementation
    """
    N = len(Y)
    return (X.T.dot(X).dot(beta) - X.T.dot(Y))/N

### Fake data
beta_true = np.array([2, 1, 3], dtype=np.float)
X = np.array([3, 1, 2]) + np.random.randn(10, 3)
Y = X.dot(beta_true) + np.random.randn(10)

### OLS gradient descent with numpy
beta_np = np.zeros(3, dtype=np.float)
for _ in range(1000):
    beta_np -= 0.1*grad_np(Y, X, beta_np)
print("beta_np: {}".format(beta_np))

# OLS gradient descent with Tensorflow
tf.reset_default_graph()
tf_X = tf.constant(X, dtype=tf.float64, name="tf_X")
tf_Y = tf.constant(Y, dtype=tf.float64, name="tf_Y")
tf_beta = tf.Variable(tf.zeros(3, dtype=tf.float64), name="beta")
tf_N = tf.cast(tf.shape(tf_Y), tf.float64)[0]
tf_Y_hat = tf.reduce_sum(tf_X*tf_beta, 1)
tf_loss = tf.reduce_sum(tf.square(tf_Y - tf_Y_hat))/(2*tf_N)
tf_gradients = tf.gradients(tf_loss, [tf_beta])[0]
# updating ops (mimic Optimizer.apply_gradients)
tf_new_beta = tf.placeholder(dtype=tf.float64, name="tf_new_beta")
tf_assign_beta = tf_beta.assign(tf_new_beta)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(tf_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
grad, _ = sess.run([tf_gradients, tf_beta])
print(np.allclose(grad_np(Y, X, np.zeros(3)), grad)) # the same as numpy implementaion

# run iteration
for _ in range(1000):
    grad, beta = sess.run([tf_gradients, tf_beta])
    beta -= 0.1*grad
    ## this is extremely slow....
    # _ = sess.run(tf_beta.assign(beta))
    ## better approach (much faster):
    _ = sess.run(tf_assign_beta, feed_dict={tf_new_beta:beta})

print("beta (with tensorflow): {}".format(beta))


tf.global_variables_initializer().run()
for _ in range(1000):
    _ = sess.run(train_op)
final_beta = sess.run(tf_beta)
print("final_beta: {}".format(final_beta))
sess.close()
