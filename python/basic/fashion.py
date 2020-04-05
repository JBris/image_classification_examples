#!/usr/bin/env python

# Source: https://www.tensorflow.org/tutorials/keras/classification
# Author: Francois Chollet - https://twitter.com/fchollet
# Data: https://github.com/zalandoresearch/fashion-mnist

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import keras.datasets.fashion_mnist as fashion_mnist

def load_data():
    return fashion_mnist.load_data()

def explore(train_images, train_labels, test_images, test_labels):
    print('Tensorflow version: ', tf.__version__)
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))

def preprocess_data(train_images, test_images):
    return (train_images / 255.0), (test_images / 255.0)

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    return model

def make_predictions(model, test_images):
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)
    print(predictions[0])
    np.argmax(predictions[0])
    img = test_images[1]
    img = (np.expand_dims(img,0))
    predictions_single = probability_model.predict(img)
    print(predictions_single)
    print(np.argmax(predictions_single[0]))

(train_images, train_labels), (test_images, test_labels) = load_data()
explore(train_images, train_labels, test_images, test_labels)
train_images, test_images = preprocess_data(train_images, test_images)
model = build_model()
model = train_model(model, train_images, train_labels)
make_predictions(model, test_images)