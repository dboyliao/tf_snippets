#!/usr/bin/env python3
# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np


def _rand_initializer(*shape):
    return np.random.randn(*shape)

def _norm_initializer(in_shape, out_shape):
    sig = np.sqrt(6.0/(in_shape+out_shape))
    return sig*(2*np.random.rand(in_shape, out_shape)-1)


def build_fc_nn(nn_arch, graph=None):
    if graph is None:
        graph = tf.Graph()

    input_size = nn_arch[0]
    with graph.as_default():
        input_layer = tf.placeholder(tf.float32,
                                     shape=[None, input_size],
                                     name="input")
        last_layer = input_layer
        for i, (in_shape, out_shape) in enumerate(zip(nn_arch[:-1], nn_arch[1:]), 1):
            with tf.name_scope("hidden_{}".format(i)):
                sig = np.sqrt(6.0/(in_shape+out_shape))
                weight = tf.Variable(_norm_initializer(in_shape, out_shape),
                                     dtype=tf.float32,
                                     name="weight")
                bias = tf.Variable(_rand_initializer(out_shape),
                                   dtype=tf.float32,
                                   name="bias")
                z_score = tf.nn.bias_add(tf.matmul(last_layer, weight), bias, name="z_score")
                last_layer = tf.nn.sigmoid(z_score)

        with tf.name_scope("output"):
            true_prob = tf.placeholder(tf.float32,
                                       shape=[None, nn_arch[-1]],
                                       name="target")
            prob = tf.nn.softmax(z_score, name="prob")
            pred = tf.arg_max(prob, 1, name="pred")
    return graph
