#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf

# tf.cond is defined in tf.python.control_flow_ops
tf.reset_default_graph()
pred = tf.placeholder(tf.bool, name="pred")
x = tf.cond(pred, 
            lambda: tf.constant(3, dtype=tf.float32), 
            lambda: tf.constant(1, dtype=tf.float32))

with tf.Session() as sess:
    print(sess.run(x, feed_dict={pred: True}))
    print(sess.run(x, feed_dict={pred:False}))
