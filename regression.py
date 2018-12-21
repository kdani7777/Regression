#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:02:07 2018

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #library for data manipulation and analysis
                    #in particular, it offers data structures and operations for manipulating
                    #numerical tables and time series

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

#shape is the number of elements in each dimension
print("Training set: {}".format(train_data.shape)) #404 examples, 13 features each
print("Testing set: {}".format(test_data.shape)) #102 examples, 13 features each

print(train_data[1])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT']

dataFrame = pd.DataFrame(train_data, columns=column_names)
print(dataFrame.head()) #shows only first 5 elements of data set

print(train_labels[1:10]) #labels are house prices in thousands

#Normalize features that use different scales and ranges
#Makes model more adaptive to changes in units used for input
mean = train_data.mean(axis=0)
std = train_data.std(axis=0) #standard deviation
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std #test data is *not* used when calculating the mean and std

print(train_data[0]) #first training sample, normalized

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
#model.summary()

#Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
#The patience parameter is the number of epochs before stopping once your loss starts
#to increase
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

#store training stats
history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2,
                    verbose=0, callbacks=[early_stopping, PrintDot()])

def plot_history(history):
    plt.figure() #only need when you want to tweak the size of the figure
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error [1000$]")
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0,5]) #limits y axis increments
    plt.show()

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True values in $1000\'s')
plt.ylabel('Predictions in $1000\'s')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100,100], [-100,100])
plt.show()

error = test_predictions - test_labels #represents an array where each element is the difference of
                                       #the respective elements in test_predictions and test_labels
plt.hist(error, bins=50)
plt.xlabel('Prediction error in $1000\'s')
plt.ylabel('Count')
plt.show()
