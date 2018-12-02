from __future__ import absolute_import, division, print_function

if False:
    get_ipython().system(' sudo -H pip install menpo opencv -q')
    get_ipython().system(' sudo -H pip install tqdm keras -q')
    get_ipython().system(' sudo -H pip install seansutils seaborn -q')
    get_ipython().system(' sudo -H pip install tensorflow-gpu -q')
    get_ipython().system(' sudo -H pip install np_utils -q')


# In[2]:


## import libaries
import pandas as pd
import numpy as np
import cv2
import os, sys
#from tqdm import tqdm

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 8)


# In[3]:


# Class for a dataset
class DataSet(object):
    """Dataset class object."""

    def __init__(self, images, labels):
        """Initialize the class."""
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels


# In[4]:


def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256,256))#use you own size
    return img


# In[5]:


def LoadData(path, validatation_data_percentage, test_data_percentage):
    assert (validatation_data_percentage>0 and validatation_data_percentage<=50), "Invalid Validation Percentage(1-50)"
    assert (test_data_percentage>0 and test_data_percentage<=50), "Invalid Test Percentage(1-50)"
    assert (validatation_data_percentage+test_data_percentage<100), "Invalid Percentages(validation+test<100)"

    classes = os.listdir(path)[:10]
    
    global n_classes
    n_classes = len(classes)
    
    '''Loading Data'''    
    n = 0
    images = []
    labels = []
    images_paths = []
    for label in classes:
        for img_file in os.listdir(path + label)[:300]:
            images_paths.append(path+label+"/"+img_file)
            labels.append(n)
        n+=1
        
    for img_path in tqdm(images_paths):
        img = read_img(path+label+"/"+img_file)
        images.append(img)

    images = np.array(images)
    labels = np.array(labels)
    labels = np.identity(len(classes))[labels]
    
    
    total_data_size = len(labels)
    '''Shuffling Data'''
    perm = np.arange(total_data_size)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    
    '''Splitting Data'''
    validation_data_size = total_data_size * validatation_data_percentage // 100
    test_data_size = total_data_size * test_data_percentage // 100
    
    test_data_images = images[:test_data_size]
    test_data_labels = labels[:test_data_size]
    
    validation_data_images = images[test_data_size:test_data_size+validation_data_size]
    validation_data_labels = labels[test_data_size:test_data_size+validation_data_size]
    
    train_data_images = images[test_data_size+validation_data_size:]
    train_data_labels = labels[test_data_size+validation_data_size:]
        
    return DataSet(train_data_images, train_data_labels), DataSet(validation_data_images, validation_data_labels), DataSet(test_data_images, test_data_labels)


# In[6]:


trainset, validationset, testset = LoadData("c:/raj/imag1/", 15,15)
print ('\t\tImages\t\t\tLabels')
print ('Training:\t', trainset.images.shape,'\t', trainset.labels.shape)
print ('Validation:\t', validationset.images.shape,'\t', validationset.labels.shape)
print ('Testing:\t', testset.images.shape,'\t', testset.labels.shape)


# # Model and parameters and Transfer learning

# In[7]:


from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# In[8]:


IMG_SIZE=256
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))


# In[10]:


add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(trainset.labels.shape[1], activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])

model.summary()


# In[11]:


batch_size = 32 
epochs = 20

train_datagen = ImageDataGenerator(
        rotation_range=0, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
train_datagen.fit(trainset.images)


# In[14]:


history = model.fit_generator(
    train_datagen.flow(trainset.images, trainset.labels, batch_size=batch_size),
    validation_data=(validationset.images, validationset.labels),
    steps_per_epoch=trainset.images.shape[0] // batch_size,
    epochs=epochs
)


# In[12]:


predictions = model.predict(testset.images)


predictions = np.argmax(predictions, axis=1)


# In[13]:


accuracy = np.mean(np.equal(predictions, np.argmax(testset.labels,1)))
print("Test Accuracy:", accuracy)


# In[46]:


model_json = history.model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
history.model.save_weights("model1.h5")
get_ipython().system(' ls')


# In[68]:



