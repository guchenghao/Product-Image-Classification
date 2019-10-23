#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\PI100Classification\PI100Classification\PI100_classification_newmodel.py
# Project: d:\PI100Classification\PI100Classification
# Created Date: Monday, January 22nd 2018, 10:11:25 am
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Thursday, 19th April 2018 3:21:19 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          ZeroPadding2D, add, concatenate)
from keras.models import Model, Sequential
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from keras.utils import np_utils, plot_model
from skimage import color, io, transform
from skimage.filters import sobel
from sklearn import preprocessing

nb_train_samples = 10000  # 训练样本数
nb_validation_samples = 2000  # 测试样本数
epochs = 80  # 迭代次数
batch_size = 20  # 25 batch的大小

train_dir = 'D:\PI100Classification\PI100Classification\PI100Dataset\PI100_train\Product_query'  # 训练集地址
test_dir = 'D:\PI100Classification\PI100Classification\PI100Dataset\PI100_test\Product_query'  # 测试集地址
all_dir = 'D:\PI100Classification\PI100Classification\PI100Dataset\PI100_all'  # 图片集地址

train_additional_datagen = ImageDataGenerator(
    zca_whitening=True, rescale=1. / 255, zoom_range=0.2)
# train_datagen = ImageDataGenerator(rescale=1. / 255)

train_additional_generator = train_additional_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical')

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(100, 100),
#     batch_size=batch_size,
#     class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical')

# 卷积神经网络模型
model = Sequential()
# CNN 1
model.add(
    Conv2D(
        64, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# CNN 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))


# CNN 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# CNN 4
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# # CNN 3
# model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))


# You must flatten the data for the dense layers
model.add(Flatten())

# Dense 1
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Dense 2
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(100, activation="softmax"))
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# plot_model(
#     model,
#     to_file='D:\PI100Classification\PI100Classification\models\model.png', show_shapes=True)  # 保存模型图片

model.save(
    'D:\PI100Classification\PI100Classification\models\origin_model_new_new_next.h5')
# 自动存储最好的模型
saveBestModel = ModelCheckpoint(
    'D:\PI100Classification\PI100Classification\models\origin_model_weights_new_new_next.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto')

# 提前结束训练
# earlyStopping = kcallbacks.EarlyStopping(
#     monitor='val_loss', patience=10, verbose=1, mode='auto')

# 验证集为整个训练集
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=train_generator,
#     validation_steps=nb_train_samples // batch_size)

# 验证集采用测试集
history = model.fit_generator(
    train_additional_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[saveBestModel],
    workers=8)

# history = model.fit_generator(
#     test_generator,
#     steps_per_epoch=nb_validation_samples // batch_size,
#     epochs=epochs,
#     validation_data=test_generator,
#     validation_steps=nb_validation_samples // batch_size,
#     callbacks=[saveBestModel],
#     workers=8)

# model.save_weights(
#     'D:\PI100Classification\PI100Classification\models\origin_model_weights_new.h5'
# )  # 保存模型的训练参数

X_epoch = np.arange(0, epochs, 1)
plt.plot(X_epoch, history.history['loss'], label='loss')
plt.plot(X_epoch, history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.title('loss&&val_loss')
plt.xlabel('epoch')
plt.ylabel('loss or val_loss')
plt.show()
