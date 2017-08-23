#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from build_model import build_fc_nn
from graph_io_utils import chkp_to_freeze_graph


def main(nn_arch, learn_rate=0.1, num_iters=1000, batch_size=200):
    """Main function"""
    print("Architecture: {}".format("x".join([str(s) for s in nn_arch])))
    mnist = read_data_sets("./mnist_data", one_hot=True)

    graph = build_fc_nn(nn_arch)
    input_layer = graph.get_tensor_by_name("input:0")
    tf_target = graph.get_tensor_by_name("output/target:0")
    tf_prob = graph.get_tensor_by_name("output/prob:0")
    tf_pred = graph.get_tensor_by_name("output/pred:0")

    with graph.as_default():
        saver = tf.train.Saver()
        loss = -tf.reduce_mean(tf_target*tf.log(tf_prob), name="loss")
        update_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, name="update_op")

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print("Variables intialized...")

        pred_feed_dict = {input_layer: mnist.test.images}
        for step in range(1, num_iters+1):
            image_batch, label_batch = mnist.train.next_batch(200)
            sess.run(update_op, feed_dict={input_layer: image_batch,
                                           tf_target: label_batch})
            if step % 100 == 0:
                pred = sess.run(tf_pred, feed_dict=pred_feed_dict)
                acc = (mnist.test.labels.argmax(axis=1) == pred).mean() * 100.0
                print("Step {}: {:.2f}%".format(step, acc))
        pred = sess.run(tf_pred, feed_dict=pred_feed_dict)
        acc = (mnist.test.labels.argmax(axis=1) == pred).mean()*100.0
        chkp_path = saver.save(sess, "train/simple_mnist-{:.2f}".format(acc), global_step=step)
    chkp_to_freeze_graph(chkp_path,
                         [tf_pred.op.name],
                         True,
                         "simple_mnist_{:.2f}.pb".format(acc))


def _arch_type(argstr):
    return [int(s) for s in argstr.strip().split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learn-rate", dest="learn_rate",
                        help="learning rate (default: 0.1)", default=0.1, type=float)
    parser.add_argument("-n", "--num-iteration", dest="num_iters",
                        help="number of iterations", default=1000, type=int)
    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        default=200, type=int,
                        help="training batch size")
    parser.add_argument("--nn-arch", dest="nn_arch",
                        type=_arch_type, required=True, metavar="M,N,K,...",
                        help="neural network architecture (seperated by ',')")
    args = vars(parser.parse_known_args()[0])
    main(**args)
