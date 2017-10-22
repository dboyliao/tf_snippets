# -*- coding:utf8 -*-
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np

__all__ = ["read_data_sets", "predict_labels"]

mnist.SOURCE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

_LABELS = { 0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"}

def predict_labels(prob):
    prob = np.atleast_2d(prob)
    pred = np.argmax(prob, axis=1)
    return np.vectorize(_LABELS.get)(pred)