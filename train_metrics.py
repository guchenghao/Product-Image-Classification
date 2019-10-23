#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\PI100Classification\PI100Classification\train_metrics.py
# Project: d:\PI100Classification\PI100Classification
# Created Date: Sunday, January 28th 2018, 5:11:43 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Thursday, 1st March 2018 9:16:07 am
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


import os
import numpy as np
import pandas as pd
# from skimage import io, color, transform
from keras.models import Sequential
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score

train_dir = 'D:\PI100Classification\PI100Classification\PI100Dataset\PI100_train\Product_query\\'  # 测试集地址

image_train = []
labels = []
train_Y = []

dirs = os.listdir(train_dir)
labels = sorted(dirs)  # 类别名

# 加载训练集图片
for label in labels:
    dirname = train_dir + label + '\\'
    for imgfile in os.listdir(dirname):
        if(imgfile[0] == '.'):
            pass
        else:
            image = load_img(dirname + imgfile, target_size=(100, 100, 3))
            image_arr = img_to_array(image)
            image_train.append(image_arr)
            train_Y.append(label)

print('训练集加载完成.....')

le = LabelEncoder()
train_Y = le.fit_transform(train_Y)

model = Sequential()

model = load_model(
    'D:\PI100Classification\PI100Classification\models\origin_model_new.h5')
model.load_weights(
    'D:\PI100Classification\PI100Classification\models\origin_model_weights_new.h5')

image_train = np.array(image_train, dtype="float") / 255.0
# image_train = np.array(image_train, dtype="float")

result = model.predict(image_train)
# print(result)
print(result.shape)

# 获取每张图片的最大预测值的那个类，并生成
submission = pd.DataFrame(result, columns=le.classes_)
# 输出2000张图片的预测值的表格
submission.to_csv(
    'D:\PI100Classification\PI100Classification\\results\PI100_classification_train_prediction_results.csv'
)
submission = submission.T
FinalSu = submission.idxmax()
FinalSu = pd.Series(FinalSu.reshape(-1, ))

# 保存模型预测分类结果
FinalSu.to_csv(
    'D:\PI100Classification\PI100Classification\\results\PI100_classification_train_Final_results.csv'
)


FinalSu = submission.idxmax().reshape(-1, )
pre_Y = le.fit_transform(FinalSu)
# print(pre_Y)

# 使用不同标准进行模型评估
PI100_f1_score = f1_score(train_Y, pre_Y, average='micro')  # f1_score
PI100_accuracy_score = accuracy_score(train_Y, pre_Y)  # accuracy_score
PI100_recall_score = recall_score(
    train_Y, pre_Y, average='micro')  # recall_score

print('f1_score:{0}'.format(PI100_f1_score))
print('accuracy_score:{0}'.format(PI100_accuracy_score))
print('recall_score:{0}'.format(PI100_recall_score))
