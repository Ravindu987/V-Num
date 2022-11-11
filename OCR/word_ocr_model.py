import cv2
import numpy as np
import string
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle

char_list = string.ascii_letters+string.digits


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst


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
    # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel = find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img


training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

annot = open('./annotation.txt',
             'r').readlines()
imagenames = []
txts = []

for cnt in annot:
    filename, txt = cnt.split(',')[0], cnt.split(',')[1].split('\n')[0]
    imagenames.append(filename)
    txts.append(txt)

c = list(zip(imagenames, txts))

random.shuffle(c)

imagenames, txts = zip(*c)


for i in range(len(imagenames)):
    img = cv2.imread(
        '../OCR train data/'+imagenames[i], 0)
    print(i)
    img = preprocess_img(img, (128, 32))
    img = np.expand_dims(img, axis=-1)
    img = img/255.
    txt = txts[i]

    # compute maximum length of the text
    if len(txt) > max_label_len:
        max_label_len = len(txt)

    # split the 150000 data into validation and training dataset as 10% and 90% respectively
    if i % 10 == 0:
        valid_orig_txt.append(txt)
        valid_label_length.append(len(txt))
        valid_input_length.append(31)
        valid_img.append(img)
        valid_txt.append(encode_to_labels(txt))
    else:
        orig_txt.append(txt)
        train_label_length.append(len(txt))
        train_input_length.append(31)
        training_img.append(img)
        training_txt.append(encode_to_labels(txt))

    # break the loop if total data is 150000
    if i == 150000:
        flag = 1
        break
    i += 1

train_padded_txt = pad_sequences(
    training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_txt = pad_sequences(
    valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

# CRNN model
inputs = Input(shape=(32, 128, 1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
print(conv_1.shape)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
print(pool_1.shape)

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
print(conv_7.shape)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list)+1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)


labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    # Defining the CTC loss.
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# CTC layer declaration using lambda.
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [outputs, labels, input_length, label_length])

# Including the CTC layer to train the model.
model = Model(inputs=[inputs, labels, input_length,
              label_length], outputs=loss_out)


model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "/best_model.hdf5"
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]


training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

print(model.summary())
batch_size = 64
epochs = 15
model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length],
          y=np.zeros(len(training_img)),
          batch_size=batch_size, epochs=epochs,
          validation_data=([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
          [np.zeros(len(valid_img))]), verbose=1, callbacks=callbacks_list)
