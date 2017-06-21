#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

tf.reset_default_graph()
with tf.variable_scope("foo"):
    w = tf.get_variable("w", shape=[3, 3])

# scope/variable_name:output_index
print(w.name)

with tf.variable_scope("foo", reuse=True):
    print(tf.get_variable("w").name)

# ## References
# - [Stackoverflow - Variable Name](https://stackoverflow.com/questions/40925652/in-tensorflow-whats-the-meaning-of-0-in-a-variables-name)
# - [Stackoverflow - Tensor Naming](https://stackoverflow.com/questions/36150834/how-does-tensorflow-name-tensors)
