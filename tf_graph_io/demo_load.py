#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from graph_io_utils import import_pb_file


tf.logging.set_verbosity(tf.logging.ERROR)


def main(pb_file):
    """Main function for demo loading trained graph
    """
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    graph = import_pb_file(pb_file, name="")
    print("graph loaded...")

    input_images = graph.get_tensor_by_name("input:0")
    predictions = graph.get_tensor_by_name("output/pred:0")

    with tf.Session(graph=graph) as sess:
        # you don't have to initialize variables here
        # since there is none.
        pred = sess.run(predictions, feed_dict={input_images: mnist.test.images})

    true_value = np.argmax(mnist.test.labels, axis=1)
    acc = (true_value == pred).mean()*100.0
    print("Accuracy on test set: {:.2f}%".format(acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pb_file", metavar="MODEL_FILE",
                        help="pre-trained protobuf file")
    args = vars(parser.parse_known_args()[0])
    main(**args)
