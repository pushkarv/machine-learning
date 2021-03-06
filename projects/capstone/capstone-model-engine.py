# coding: utf-8

# # Distracted Driving Detection

import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import datetime, random, pickle
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
import cv2
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ActivityRegularization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras import regularizers
import sys
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

############ CONFIGURABLE PARAMETERS ########################
#location to save models
file_root='saved_models/'
# prefix used for saving the model and history files
prefix_str = str(datetime.date.today()) + str(random.randint(1, 100))

# Number of epochs to perform training for
NUM_EPOCHS = 5
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Location of the image files
images_path = "sample_images/train"

#############################################################


# dictionary for distraction category to numerical value
catLabels = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
}

#Maps category value to categorical label in the catLabels object
def getClass(value):
    index = 'c' + str(value)
    return catLabels[index]


###  Creates multiple models ####
# All models here use the Keras API to create a Sequential model - essentially a stack of layers
## The default kernel size of 4x4 is used, and all filters of size 10 are used for Conv2D transformations

#This model consists of a Convolution 2D layer, 2D Max Pooling layer, a Global averaging layer and a Dense layer
#with a softmax activation function
def create_base_model():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Base Model")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model consists of a Convolution 2D layer, Batch Normalization layer, 2D Max Pooling layer,
# a Global averaging layer and a Dense layer #with a softmax activation function
def create_model1():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model1 with Batch Normalization")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


#This model consists of 2 Convolution 2D layers, 2 2D Max Pooling layers,
# a Global averaging pooling layer and a Dense layer with a softmax activation function
def create_model2():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model2 - Multiple conv2d layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


#This model consists of 2 Convolution 2D layers, 2 2D Max Pooling layers,
# a Global averaging pooling layer and a Dense layer with a softmax activation function
def create_model3():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model3 - Multiple conv2d layers w/ Batch Normalization for each Conv2D")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model adds a BatchNormalization layers between the Conv2D and MaxPooling 2D layers.
def create_model4():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print("Model4 - Multiple conv2d layers w/ Batch Normalization for each Conv2D + 2 softmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model has Batch Normalization between Conv2D and MaxPooling 2D layers, but it also has
# Dropout layers between the Dense (fully connected) layers and an additional Dense layer than the model above.
def create_model5():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax'))
    print("Model5 - Multiple conv2d layers w/ Batch Normalization for each Conv2D + 3 softmax layers + Dropouts")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model further adds Dropouts along with the BatchNormalization layers between the Conv2D and MaxPooling 2D layers
def create_model6():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.15))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax'))
    print(
    "Model6 - Dropouts + Multiple conv2d layers w/ Batch Normalization for each Conv2D + 3 softmax layers + Dropouts")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model use 2 sets of repeating Conv2D/MaxPooling2D layers, and a Dropout layers for each Dense layer.
def create_model7():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    print(
            sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 softmax layers + Dropouts + activity_regularizers in dense layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model adds an additional Conv2D/MaxPooling2D layer and an additional set of Dropout/Dense layers.
#This also combines the L1 Regularization and Dropout techniques
def create_model8():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    print(
            sys._getframe().f_code.co_name + " - Multiple conv2d layers w/ for each Conv2D + 3 softmax layers + Dropouts + kernel_regularizers in dense layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is similar to model8, but use different values for the Dropout - essentially adjusts the hyperparameter.
def create_model9():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.15))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(Dropout(.20))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax'))
    print(
            sys._getframe().f_code.co_name + " - Multiple conv2d layers w/ Batch Normalization for each Conv2D + 3 softmax layers + Dropouts")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is has only 3 sets of Conv2D /MaxPooling2D layers and 3 Dense layers with no regularization.
def create_model10():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is similar to model10, but less Dense layers
def create_model11():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is similar to model10, but more Dense layers.
def create_model12():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is similar to model12, but only 1 set of Conv2D/MaxPooling2D layers.
def create_model13():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This model is similar to model10, but it adds the L1 regularizers
def create_model14():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model14, but adjusts the value of the L1 Regularizer that only exists on a single layer
def create_model15():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(.50)))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model15, but moves the position of the L1 Regularizer
def create_model16():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


## DUPLICATE OF MODEL 16 - IGNORING RESULTS ###
def create_model17():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model 14 with different L1 Regularizer value
def create_model18():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.1)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.1)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.1)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


#This is similar to model 14 with different L1 Regularizer value
def create_model19():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.25)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.25)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.25)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model19, but less Dense layers
def create_model20():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model 14 with different L1 Regularizer values
def create_model21():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    sgd = optimizers.SGD(lr=0.10, decay=.01, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

#This is similar to model 14 with different L1 Regularizer values
def create_model22():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.30)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


## Better models
#This model has the same 3 set of Conv2D/MaxPooling2D layers, but it uses an additional Flatten layer that creates an
# (MxNxP)x1 tensor from a MxNxP tensor.
def create_model23(dropout):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - 3 conv2d layers + 2 dense  layers, relu + softmax and dropout = " + str(
        dropout))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name + 'dropout_' + str(dropout)}

#This is similar to the model23, but uses grayscale  - structurally same as model23, but use separately for tracking
#experiment results
def create_model23_grayscale(dropout):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - 3 conv2d layers + 2 dense  layers, relu + softmax and dropout = " + str(
        dropout))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name + 'dropout_' + str(dropout)}

#This is similar to model23 but add a L1 regularizer on the Dense layer with the relu function
def create_model24(regularizer_value):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu', activity_regularizer=regularizers.l1(regularizer_value)))
    model.add(Dropout(.20))
    model.add(Dense(units=10, activation='softmax'))
    print(
            sys._getframe().f_code.co_name + " - 3 conv2d layers + 2 dense  layers, relu + softmax and regularizer = " + str(
        regularizer_value))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name + 'regularizer_' + str(regularizer_value)}

#This model removes the Dropout layer from model24
def create_model25(regularizer_value):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(units=10, activation='relu', activity_regularizer=regularizers.l1(regularizer_value)))
    model.add(Dense(units=10, activation='softmax'))
    print(
            sys._getframe().f_code.co_name + " - 3 conv2d layers + 2 dense  layers, relu + softmax and regularizer = " + str(
        regularizer_value))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name + 'regularizer_' + str(regularizer_value)}

#This model has both Dense layers with a softmax function, and has Dropout in between with a Flatten layer before.
def create_model26(dropout):
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(dropout))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - 3 conv2d layers + 2 dense  layers, relu + softmax and dropout = " + str(
        dropout))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name + 'dropout_' + str(dropout)}


# Load all the image data from the path provided
def loadImages(path):
    data = load_files(path)
    files = data['filenames']
    targets = data['target']
    target_names = data['target_names']
    return files, targets, target_names


# Resize image to 224x224
# Convert image to an array -> resized to a 4D tensor used by Keras CNN
# Tensor will be (1,224,224,3)
# Adopted from the Deep Learning Project
def path_to_tensor(img_path, equalized=False):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224), grayscale=equalized)
    if (equalized == True):
        img = np.array(img)
        #equalize histogram
        img = equalize_histogram(img, 16)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# Iterates over all images and convert to tensor
def paths_to_tensor(img_paths):
    print (img_paths)
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# Evaluates the model accuracy using the Test dataset
def predict_distraction(model):
    print("Evaluating...")
    scores = model.evaluate(test_tensors, test_targets_onehot, verbose=0)
    print("Evaluation %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    return

# Saves the accuracy / loss curves from the trained model history data
def plot_learning_history(m):
    history = m['history']
    # history for accuracy

    fig, ax = plt.subplots()
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    plt.title(m['model_name'] + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    prefix_str = m['model_name'] + str(datetime.date.today()) + str(random.randint(1, 100))
    fig.savefig(file_root + prefix_str + '_model_accuracy.png')
    plt.close(fig)

    #  history for loss
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title(m['model_name'] + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig.savefig(file_root + prefix_str + '_model_loss.png', format="png")
    plt.close(fig)

    print("Model: " + m['model_name'])
    predict_distraction(m['model'])
    return

# Performs the model training with a validation dataset split based on validation_split parameter
def train_model(_epochs, _model):
    history = _model.fit(train_tensors, train_targets_onehot, validation_split=.25,
                         epochs=_epochs, batch_size=32, callbacks=[], verbose=2)
    # for layer in _model.layers:
    #     g = layer.get_config()
    #     h = layer.get_weights()
    #     print (g)
    #     print (h)
    return history

# Plots a histogram of an image - this requires the image to be grayscale
def plot_histogram(image):
    hist,bins = np.histogram(image.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    return

# Equalizes the histogram of an image based on a kernel size - requires grayscale image
def equalize_histogram(img, grid_size):
    clahe = cv2.createCLAHE(tileGridSize=(grid_size,grid_size))
    equalized_img = clahe.apply(img)
    return equalized_img


#################### MAIN PROGRAM #######################

#Initialize Tensorflow to make sure Tensorflow is installed properly
#If GPU libraries are available, GPU will be used by default in Tensorflow
hello = tf.constant('Hello, TensorFlow works!')
sess = tf.Session()
print(sess.run(hello))

# ## Load the Data

print ("Loading Images...")

files, targets, target_names = loadImages(images_path)
# predict_files = np.array(glob("images/test/*"))[1:10]
print('Number of Categories: ', len(target_names))
print('Categories: ', target_names)
print('Number of images by category: ')
for c in target_names:
    print(c + ':' + str(len(os.listdir(images_path + '/' + c))))

# Split the original training sets into training & testing sets
train_files, test_files, train_targets, test_targets = train_test_split(files, targets, test_size=0.20, random_state=40)

print(train_files.shape, test_files.shape, train_targets.shape, test_targets.shape)
print(len(test_files))

print ("Creating image tensors")

train_tensors = paths_to_tensor(train_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

print("Size of train tensors: " + str(train_tensors.shape))
print("Size of test tensors: " + str(test_tensors.shape))
print("Size of test targets: " + str(test_targets.shape))

# predict_tensors = paths_to_tensor(predict_files).astype('float32')/255

print("Train Targets", train_targets)
print ("Test Targets", test_targets)
train_targets_onehot = np_utils.to_categorical(np.array(train_targets), 10)
test_targets_onehot = np_utils.to_categorical(np.array(test_targets), 10)
print ("Train Targets One-hot encoded", train_targets_onehot)
print ("Test Targets One-hot encoded", test_targets_onehot)

print(train_targets_onehot.shape)
print(test_targets_onehot.shape)

print ("Selecting models for training")

dropout_values = [.05, .10, .15, .20, .25, .30, .35, .40]
regularizer_values = [.05, .10, .15, .20, .25, .30, .40, .50, .60]


models = []

# for r in regularizer_values:
#     models.extend([create_model25(r)])

#for d in dropout_values:
models.extend([create_base_model()])

print ("Number of Epochs: ", NUM_EPOCHS)
print ("Training " + str(len(models)) + " models")

# Prints all the models that are about to be trained
for m in models:
    print (m)

# Iterate through all models and perform training and plot the model accuracy curves
for m in models:
    print ("Training Model: ", m['model_name'])
    prefix_str = m['model_name'] + str(datetime.date.today()) + str(random.randint(1, 100))
    # checkpointer = ModelCheckpoint(filepath=file_root + prefix_str + '_model.best.from_scratch.hdf5',
    #                          verbose=1, save_best_only=True)
    m['history'] = train_model(NUM_EPOCHS, m['model'])
    m['model'].save(file_root + prefix_str + '_complete_model.hdf5')
    with open(file_root + prefix_str + '_trainHistoryDict', 'wb') as file_pi:
        pickle.dump(m['history'].history, file_pi)
    plot_learning_history(m)


