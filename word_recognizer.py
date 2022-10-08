import fnmatch
import cv2
import numpy as np
import string
import time
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle


def find_dominant_color(image):
    # Resizing parameters
    width, height = 150, 150
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


def preprocess_img(img, imgSize):
    "put img into target img of size imgSize and normalize gray-values"

    # In case of black images with no text just use black image instead.
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    # INTER_CUBIC interpolation best approximate the pixels image
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("test2", img)
    # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel = find_dominant_color(Image.fromarray(img))
    # cv2.imshow("test3", img)
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img
    # cv2.imshow("test4", img)

    img = target
    # cv2.imshow("test5", img)

    return img


char_list = string.ascii_letters+string.digits

# CRNN model
inputs = Input(shape=(32, 128, 1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list)+1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)

annot = open('./annotation.txt', "r").readlines()
images = []
imagenames = []
txts = []

# for i in range(10):
#     filename, txt = annot[i].split(',')[0], annot[i].split(',')[
#         1].split('\n')[0]
#     imagenames.append(filename)
#     txts.append(txt)

#     img = cv2.imread("./wordimages/"+filename, 0)
#     img = preprocess_img(img, (128, 32))
#     img = np.expand_dims(img, axis=-1)
#     img = img/255
#     images.append(img)

for filename in os.listdir('./Sample_DataSet'):
    txts.append(filename)
    img = cv2.imread("./Sample_DataSet/"+filename, 0)
    # cv2.imshow("Test", img)
    img = preprocess_img(img, (128, 32))
    # cv2.imshow("Image", img)
    img = np.expand_dims(img, axis=-1)
    img = img/255
    images.append(img)

    cv2.waitKey(0)


act_model.load_weights("./Trained Weights/best_model.hdf5")
stacked_imgs = np.stack(images)
prediction = act_model.predict(stacked_imgs)

out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                               greedy=True)[0][0])

i = 0
for x in out:
    print("Original Text: " + txts[i])
    print("Prediction: ")
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end='')
    print('\n')
    i += 1
