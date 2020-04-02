#!/usr/bin/env python

# Source: https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
# Author: Pulkit Sharma - https://www.analyticsvidhya.com/blog/author/pulkits/

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

def load_data():
    dirname = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(dirname + '/data/train.csv')

def split_data(train):
    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)
    y=train['label'].values
    y = to_categorical(y)

    return = train_test_split(X, y, random_state=42, test_size=0.2)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

train = load_data()
X_train, X_test, y_train, y_test = split_data(train)
model = build_model()
train_model(model, X_train, X_test, y_train, y_test)
