#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
# shut up Cpp logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' 

def leaky_relu(_x, alpha=0.1, name=None):
    if name is None:
        name = "leaky_relu"
    dtype = _x.dtype
    _alpha = tf.get_variable(name,
                             dtype=dtype,
                             shape=_x.get_shape(),
                             initializer=tf.constant_initializer(0.1))
    return tf.maximum(_alpha*_x, _x)


def leaky_relu_numpy(x, alpha=0.1):
    ret = x.copy()
    ret[np.where(x < 0)] *= alpha
    return ret


arr = np.array([-1, 2, 3], dtype=np.float)
print(leaky_relu_numpy(arr))


tf.reset_default_graph()
with tf.variable_scope("test"):
    tf_x = tf.constant(arr, name="tf_x")
    tf_leakyReLU = leaky_relu(tf_x)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


print(sess.run(tf_leakyReLU))

# - [StackOverflow](https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow)



