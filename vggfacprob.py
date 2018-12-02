from keras.layers import Activation, Dense
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import Model
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
import cv2
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, 7, 7, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, 1, 1, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, 1, 1))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')
def preprocess_image(image_path):
    #img = load_img(image_path, target_size=(224, 224))
    #img = cv2.imread(image_path)
    #img = cv2.resize(img, (224, 224))
    #img = img_to_array(img)
    #img=np.asarray(img)
    #img = np.expand_dims(img, axis=0)
    #img = preprocess_input(img)
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, data_format = 'channels_first',version=1)
    return x
    #return img
vgg_face_descriptor = Model(input=model.layers[0].input, output=model.layers[-2].output)
img1_representation = vgg_face_descriptor.predict(preprocess_image('1.jpg'))[0,:]
img2_representation = vgg_face_descriptor.predict(preprocess_image('2.jpg'))[0,:]

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
epsilon = 0.40 #cosine similarity
#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
 img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
 img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
 
 cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
 euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
 
 if(cosine_similarity < epsilon):
  print("verified... they are same person")
 else:
  print("unverified! they are not same person!")


vi /usr/local/lib/python2.7/dist-packages/keras_vggface/utils.py
vi /usr/local/lib/python2.7/dist-packages/keras_vggface/models.py
pip install keras_vggface
 pip install keras_vggface --no-dependencies
pip install pillow

vi /usr/local/lib/python2.7/dist-packages/keras_vggface/models.py
replace( problem with keras 2.2.2
from keras.applications.imagenet_utils import _obtain_input_shape (this)
with
from keras_applications.imagenet_utils import _obtain_input_shape
ImportError: cannot import name _obtain_input_shape

from keras_applications.imagenet_utils import _obtain_input_shape

softmax axis prob
tried to upgrade tensorflow
used only pip instead
sudo pip3 install --upgrade tensorflow