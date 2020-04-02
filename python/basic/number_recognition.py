#!/usr/bin/env python

# Source: https://www.blog.duomly.com/how-to-create-image-recognition-with-python/

# 1. Import modules, classes and functions
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

def main():
    # 2. Load data
    x, y = load_digits(n_class=10, return_X_y=True)

    # 3. Transform and split data
    # Create the binary output
    tr = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    y = tr.fit_transform(y)
    # Split train and test data
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    # Standardize the input
    sc = StandardScaler()
    x_train, x_test = sc.fit_transform(x_train), sc.transform(x_test)

    # 4. Create the classification model and train (fit) it
    cl = Sequential()
    # Add the hidden layer
    cl.add(Dense(units=500, activation='relu', use_bias=True,
                kernel_initializer='uniform', bias_initializer='zeros',
                input_shape=(x_train.shape[1],)))
    # Add the output layer
    cl.add(Dense(units=10, activation='softmax', use_bias=True,
                kernel_initializer='uniform', bias_initializer='zeros'))
    # Compile the classification model
    cl.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=['accuracy'])
    # Fit (train) the classification model
    cl.fit(x_train, y_train, epochs=100, batch_size=10)

    # 5. Test the classification model
    result = cl.evaluate(x_test, y_test, batch_size=128)
    for i in range(2):
        print(f'{cl.metrics_names[i]}: {result[i]}')

main()