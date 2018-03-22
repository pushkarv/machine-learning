
# coding: utf-8

# # Distracted Driving Detection

# ## Load the Data

# In[1]:

print("Starting Model Learning using multiple models to determine best model")

NUM_EPOCHS = 250

print("Number of Epochs: ", NUM_EPOCHS)

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

def getClass(value):
    index = 'c' + str(value)
    return catLabels[index]


# In[2]:


from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# import tensorflow as tf
# from keras import backend as K

# num_cores = 4
# GPU = 1
# CPU = 0

# if GPU:
#     num_GPU = 1
#     num_CPU = 1
# if CPU:
#     num_CPU = 1
#     num_GPU = 0

# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#         device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
# session = tf.Session(config=config)
# K.set_session(session)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print ("Loading Images...")

def loadImages(path):
        data = load_files(path)
        files = data['filenames']
        targets = data['target']
        target_names = data['target_names']
        return files, targets, target_names

path = "images/train"
files,targets,target_names = loadImages(path)
predict_files = np.array(glob("images/test/*"))[1:10]
print('Number of Categories: ', len(target_names))
print('Categories: ', target_names)
print('Number of images by category: ')
for c in target_names:
    print(c + ':' + str(len( os.listdir(path+'/'+c))))
    # train_data = np.vstack((files, targets)).T
    # print(train_data.shape)

#Split the original training sets into training & validation sets
train_files, test_files, train_targets, test_targets = train_test_split(files, targets, test_size=0.20, random_state=40)

print(train_files.shape, test_files.shape, train_targets.shape, test_targets.shape)
print(len(test_files))


# # Data Analysis

# In[3]:


import cv2
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

def displayImage(sample_image):
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    #plt.imshow(cv_rgb)
    #plt.show()

for i in range(1,5):
    sample_image = cv2.imread(train_files[i])
    print(train_targets[i])
    print(getClass(train_targets[i]))
    displayImage(sample_image)



# In[4]:



#(nb_samples,rows,columns,channels)
#nb_samples - total number of images
# Resize image to 224x224
# Convert image to an array -> resized to a 4D tensor used by Keras CNN
# Tensor will be (1,224,224,3)

#Adopted from the Deep Learning Project
from keras.preprocessing import image
from tqdm import tqdm

print ("Creating image tensors")

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


#
# ## Pre-Process the Data
#

# In[5]:


#Rescale the images

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
#predict_tensors = paths_to_tensor(predict_files).astype('float32')/255


# ## Baseline Model Architecture

# In[56]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, ActivityRegularization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras import regularizers
import sys

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

models = [create_base_model(), create_model1(), create_model2(), create_model3(),
         create_model4(), create_model5(), create_model6() ]


for m in models:
    print (m)
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))


# ## Train the Model

# In[61]:


from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import datetime, random, pickle

print("Train Targets", train_targets)
print ("Test Targets", test_targets)
train_targets_onehot = np_utils.to_categorical(np.array(train_targets),10)
test_targets_onehot = np_utils.to_categorical(np.array(test_targets),10)
print ("Train Targets One-hot encoded", train_targets_onehot)
print ("Test Targets One-hot encoded", test_targets_onehot)

print(train_targets_onehot.shape)
print(test_targets_onehot.shape)

file_root='C:/Users/pushkar/ML/machine-learning/projects/capstone/saved_models/'
prefix_str = str(datetime.date.today()) + str(random.randint(1,100))

print ("Number of Epochs: ", NUM_EPOCHS)

def train_model(_epochs, _model):
    history = _model.fit(train_tensors, train_targets_onehot, validation_split=.20,
          epochs=_epochs, batch_size=32, callbacks=[checkpointer], verbose=2)
    return history

for m in models:
    print ("Training Model: ", m['model_name'])
    prefix_str = m['model_name'] + str(datetime.date.today()) + str(random.randint(1,100))
    checkpointer = ModelCheckpoint(filepath=file_root + prefix_str + '_model.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)
    m['history'] = train_model(NUM_EPOCHS,m['model'])
    with open(file_root + prefix_str + '_trainHistoryDict', 'wb') as file_pi:
        pickle.dump(m['history'].history, file_pi)




# In[69]:


import matplotlib.pyplot as plt
import numpy as py

def predict_distraction(model):
    # get index of predicted distraction for each image in test set
    distraction_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(distraction_predictions)==np.argmax(test_targets, axis=0))/len(distraction_predictions)
    return test_accuracy


for m in models:
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


#    p = m.predict(test_tensors)
#print (p)
#z=np.argmax(p,axis=1)
#print("z = ", z)
# for i in range(1,15):
#     img = np.squeeze(np.array(test_tensors[i]))
#     displayImage(img)
#     print("Predicted class", getClass(z[i]))
#     print ("Actual Class", getClass(test_targets[i]))
