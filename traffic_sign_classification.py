#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dataset available at: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
#Library import
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import datetime
import time
import os
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore') # filter warnings

import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system('rm -rf ./logs/ # Clear any logs from previous runs')


# ## 1.1 German Data Loading

# In[ ]:


DATADIR = "GTSRB/Final_Training/Images" # Setting the directory folder

CATEGORIES = [] #label list
for i in range(43): # populate label list
    CATEGORIES.append(f"{i:05d}") # keep zeros before number

german_dataset = []
IMG_SIZE = 100

def create_dataset_german():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  # create path to image folder
        class_num = CATEGORIES.index(category)  # get the index for classification  (0 = 00000, 1 = 00001 ...)

        for img in tqdm(os.listdir(path)):  # iterate over each image of the classes
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to harmonize data size
                german_dataset.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass
create_dataset_german()

# Shuffle the dataset
random.shuffle(german_dataset)

# Create X (features) and y (labels) set 
X_ger = []
y_ger = []

for features,label in german_dataset:
    X_ger.append(features)
    y_ger.append(label)


# ## 1.2 Data augmentation on German Dataset

# In[ ]:


def data_augment(image):
    rows= image.shape[0]
    cols = image.shape[1]
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),10,1) # Image rotation
    M_trans = np.float32([[1,0,3],[0,1,6]]) # Image Translation
    img = cv2.warpAffine(image,M_rot,(cols,rows))
    img = cv2.warpAffine(img,M_trans,(cols,rows))
    img = cv2.bilateralFilter(img,9,75,75) # Bilateral filtering
    return img

classes = 43

X_full_ger = X_ger
y_full_ger = y_ger
X_aug_1 = []
Y_aug_1 = []

for i in range(0,classes):
    class_records = np.where(y_ger==i)[0].size
    max_records = 2500
    if class_records != max_records:
        ovr_sample = max_records - class_records
        samples = X[np.where(y_ger==i)[0]]
        X_aug = []
        Y_aug = [i] * ovr_sample
        
        for x in range(ovr_sample):
            img = samples[x % class_records]
            trans_img = data_augment(img)
            X_aug.append(trans_img)
            
        X_full_ger = np.concatenate((X_full_ger, X_aug), axis=0)
        y_full_ger = np.concatenate((y_full_ger, Y_aug)) 
        
        Y_aug_1 = Y_aug_1 + Y_aug
        X_aug_1 = X_aug_1 + X_aug


# In[ ]:


# Convert list to numpy arrays for further manipulation
X_full_ger = np.array(X_full_ger)
y_full_ger = np.array(y_full_ger)

# Data normalization
X_full_ger = X_full_ger/255.0

y_full_ger = np.asarray(y_full_ger).astype('float32')
n_classes = len(np.unique(y_full_ger))

Y_GER_FULL = to_categorical(y_full_ger, n_classes)
X_GER_FULL = np.array(X_full_ger).reshape(-1, 100, 100, 1)


# ## 2.1 Danish data loading

# In[ ]:


DATADIR = "DANISH_TRAFFIC_SIGNS_TEST"

#DATADIR = "/content/drive/MyDrive/ML_Project/ML Final Project/Images"

CATEGORIES = ["00003", "00013", "00014", "00017", "00022", "00024", "00025", "00028", "00029", "00038"] #label list


dataset_danish = []
IMG_SIZE = 100

def create_dataset_danish():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  # create path to image folder
        class_num = int(category)  # get the index for classification  (0 = 00000, 1 = 00001 ...)

        for img in tqdm(os.listdir(path)):  # iterate over each image of the classes
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                dataset_danish.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass
create_dataset_danish()

print(len(dataset_danish))

# Shuffle the dataset
random.shuffle(dataset_danish)

# Create X (features) and y (labels) set 
X_dan = []
y_dan = []

for features,label in dataset_danish:
    X_dan.append(features)
    y_dan.append(label)


# ## 2.2 Data augmentation on Danish Dataset

# In[ ]:


def data_augment(image):
    rows= image.shape[0]
    cols = image.shape[1]
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),10,1) # Image rotation
    M_trans = np.float32([[1,0,3],[0,1,6]]) # Image Translation
    img = cv2.warpAffine(image,M_rot,(cols,rows))
    img = cv2.warpAffine(img,M_trans,(cols,rows))
    img = cv2.bilateralFilter(img,9,75,75) # Bilateral filtering
    return img

classes = ["00003", "00013", "00014", "00017", "00022", "00024", "00025", "00028", "00029", "00038"] #label list

#X_dan = X_dan
#y_dan = y_dan
X_aug_1 = []
Y_aug_1 = []

for class_dan in classes:
    
    class_records = np.where(y_dan==int(class_dan))[0].size
    max_records = 100
    if class_records != max_records:
        ovr_sample = max_records - class_records
        samples = X[np.where(y_dan==int(class_dan))[0]]
        X_aug = []
        Y_aug = [int(class_dan)] * ovr_sample
        
        for x in range(ovr_sample):
            img = samples[x % class_records]
            trans_img = data_augment(img)
            X_aug.append(trans_img)
            
        X_dan = np.concatenate((X_dan, X_aug), axis=0)
        y_dan = np.concatenate((y_dan, Y_aug)) 
        
        Y_aug_1 = Y_aug_1 + Y_aug
        X_aug_1 = X_aug_1 + X_aug


# In[ ]:


# Convert list to numpy arrays for further manipulation
X_dan = np.array(X_dan)
y_dan = np.array(y_dan)

# Data normalization
X_dan = X_dan/255.0

y_dan = np.asarray(y_dan).astype('float32')
n_classes = len(np.unique(y_dan))

Y_DAN_TEST = to_categorical(y_dan, n_classes)
X_DAN_TEST = np.array(X_dan).reshape(-1, 100, 100, 1)


# ## 3 Final pre-processing

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_GER_TRAIN, X_GER_TEST, Y_GER_TRAIN, Y_GER_TEST = train_test_split(X_GER_FULL, Y_GER_FULL, shuffle=True)


# In[ ]:


# Shuffle the Danish (test) data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X_DAN_TEST, Y_DAN_TEST = unison_shuffled_copies(X_DAN_TEST, Y_DAN_TEST)


# ## 4.1 AlexNet Architecture

# In[ ]:


model_alexnet = Sequential([
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model_alexnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_alexnet.fit(X_ger_train,
          y_ger_train, 
          batch_size=128, 
          epochs=10, #100
          verbose=1,
          validation_split=0.2
         )
model_alexnet.save("models/alexnet_model")


# ## 4.2 Tensorboard configuration

# In[ ]:


dense_layers = [2, 1]
layer_sizes = [32, 64, 128]
conv_layers = [2, 3, 1]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, kernel_size=(3, 3),activation='relu', input_shape=X.shape[1:]))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size*2, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(BatchNormalization())
                model.add(Dropout(0.25))
                
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
              
            for n in range(dense_layer-1):
                model.add(Dense(layer_size, activation='relu'))
                model.add(Dropout(0.5))
                
            model.add(Dense(n_classes, activation='softmax'))
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME),
                                      write_graph=True,
                                      write_grads=True,
                                      write_images=True)

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            model.summary()
            model.fit(X_train,
                      y_train, 
                      batch_size=128, 
                      epochs=10, #100
                      verbose=1,
                      callbacks=[tensorboard],
                      validation_split=0.2
                     )
            model.save("models/{}".format(NAME))


# ## 5.1 Testing

# In[ ]:


# Loading the best 3 models and the Alexnet 
# NOTE: the models created with the Tensorboard architecture 
# will have unique name so the path will need to be changed 
model_3c_128n_2d = tf.keras.models.load_model('models/3-conv-128-nodes-2-dense-1653013127') 
model_3c_64n_2d = tf.keras.models.load_model('models/3-conv-64-nodes-2-dense-1653002660')
model_3c_64n_1d = tf.keras.models.load_model('models/3-conv-64-nodes-1-dense-1653028639')
model_alexnet = tf.keras.models.load_model('models/alexnet_model')


# In[ ]:


def testing(model, X_test, y_test):
    # Testing our model on Danish Test Data
    # Convert one-hot to index
    y_true = np.argmax(y_test, axis=1) 
    # Make prediction
    y_pred = model.predict(X_test)
    # Convert one-hot to index
    y_pred = np.argmax(y_pred, axis=1) 
    # Classification report
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.set(rc = {'figure.figsize':(18,8)})
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    
    print(y_true, y_pred)


# In[ ]:


# German data testing 
testing(model_3c_128n_2d, X_GER_TEST, Y_GER_TEST)
testing(model_3c_64n_2d, X_GER_TEST, Y_GER_TEST)
testing(model_3c_64n_1d, X_GER_TEST, Y_GER_TEST)
testing(model_alexnet, X_GER_TEST, Y_GER_TEST)

# Danish data testing 
testing(model_3c_128n_2d, X_DAN_TEST, Y_DAN_TEST)
testing(model_3c_64n_2d, X_DAN_TEST, Y_DAN_TEST)
testing(model_3c_64n_1d, X_DAN_TEST, Y_DAN_TEST)
testing(model_alexnet, X_DAN_TEST, Y_DAN_TEST)

