import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random


def visualize(img, seg_img):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.title('Segmented Image')
    plt.show()


def get_segmented_img(img, n_classes):
    seg_labels = np.zeros((512, 512, 1))
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]

    seg_labels[:, :, 0] = (img != 0).astype(int)

    return seg_labels


def preprocess_img(img):
    img = cv2.resize(img, (512, 512))
    return img


def batch_generator(filelist, n_classes, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'./PageSegData/PageImg/{fn}.JPG', 0)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis=-1)
            img = img/255

            seg = cv2.imread(f'./PageSegData/PageSeg/{fn}_mask.png', 1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# model = FCN(n_classes=2,
#  input_height=320,
#  input_width=320)
model = unet()

image_list = os.listdir('./PageSegData/PageImg/')
image_list = [filename.split(".")[0] for filename in image_list]

file_train = image_list[0:int(0.75*len(image_list))]
file_test = image_list[int(0.75*len(image_list)):]

mc = ModelCheckpoint('weights{epoch:08d}.h5',
                     save_weights_only=True, period=1)
model.fit_generator(batch_generator(file_train, 2, 2), epochs=3, steps_per_epoch=1000, validation_data=batch_generator(file_test, 2, 2),
                    validation_steps=400, callbacks=[mc], shuffle=1)
