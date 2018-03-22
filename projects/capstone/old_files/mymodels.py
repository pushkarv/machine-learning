
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
import mymodels

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ActivityRegularization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras import regularizers
import sys
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

ImageFile.LOAD_TRUNCATED_IMAGES = True
NUM_EPOCHS = 250

print("Number of Epochs: ", NUM_EPOCHS)

prefix_str = str(datetime.date.today()) + str(random.randint(1,100))

#dictionary for distraction category to numerical value
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

def loadImages(path):
        data = load_files(path)
        files = data['filenames']
        targets = data['target']
        target_names = data['target_names']
        return files, targets, target_names

def displayImage(sample_image):
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    #plt.imshow(cv_rgb)
    #plt.show()

#Adopted from the Deep Learning Project

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    print (img_paths)
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
	
def getClass(value):
    index = 'c' + str(value)
    return catLabels[index]

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
    #plt.show()
    prefix_str = m['model_name'] + str(datetime.date.today()) + str(random.randint(1,100))
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
    #plt.show()
    fig.savefig(file_root + prefix_str + '_model_loss.png', format="png")
    plt.close(fig)

    print("Model: " + m['model_name'])
    test_accuracy = predict_distraction(m['model'])
    print('Test accuracy: %.4f%%' % test_accuracy)

	
print ("Creating Models")

def create_base_model():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Base Model")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model1():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model1 with Batch Normalization")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model2():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model2 - Multiple conv2d layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model3():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    print("Model3 - Multiple conv2d layers w/ Batch Normalization for each Conv2D")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model4():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print("Model4 - Multiple conv2d layers w/ Batch Normalization for each Conv2D + 2 softmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model5():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
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

def create_model6():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(Dropout(.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(Dropout(.15))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
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
    print("Model6 - Dropouts + Multiple conv2d layers w/ Batch Normalization for each Conv2D + 3 softmax layers + Dropouts")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model7():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 softmax layers + Dropouts + activity_regularizers in dense layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model8():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers w/ for each Conv2D + 3 softmax layers + Dropouts + kernel_regularizers in dense layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model9():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(Dropout(.1))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(Dropout(.15))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(Dropout(.20))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(.1))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.2))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dropout(.3))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers w/ Batch Normalization for each Conv2D + 3 softmax layers + Dropouts")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model10():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model11():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 2 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model12():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
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

def create_model13():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
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

def create_model14():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model15():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model16():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model17():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.01)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}

def create_model18():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.1)))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.1)))
    model.add(Dense(units=10, activation='softmax',activity_regularizer=regularizers.l1(0.1)))
    print(sys._getframe().f_code.co_name + " - Multiple conv2d layers + 3 dense sofmax layers")
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return {"model": model, "model_name": sys._getframe().f_code.co_name}


def predict_distraction(model):
    # get index of predicted distraction for each image in test set
    distraction_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(distraction_predictions)==np.argmax(test_targets, axis=0))/len(distraction_predictions)
    return test_accuracy

	
def train_model(_epochs, _model):
    history = _model.fit(train_tensors, train_targets_onehot, validation_split=.20,
          epochs=_epochs, batch_size=32, callbacks=[checkpointer], verbose=2)
    return history

def start_training(models):
	for m in models:
		print ("Training Model: ", m['model_name'])
		prefix_str = m['model_name'] + str(datetime.date.today()) + str(random.randint(1,100))
		checkpointer = ModelCheckpoint(filepath=file_root + prefix_str + '_model.best.from_scratch.hdf5',
								   verbose=1, save_best_only=True)
		m['history'] = train_model(NUM_EPOCHS,m['model'])
		with open(file_root + prefix_str + '_trainHistoryDict', 'wb') as file_pi:
			pickle.dump(m['history'].history, file_pi)
		plot_learning_history(m)
