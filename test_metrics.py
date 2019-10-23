#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\PI100Classification\PI100Classification\test_metrics.py
# Project: d:\PI100Classification\PI100Classification
# Created Date: Saturday, January 6th 2018, 7:51:42 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Thursday, 31st May 2018 5:05:08 pm
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
from sklearn.metrics import f1_score, accuracy_score, recall_score, mean_squared_error

test_dir = 'D:\PI100Classification\PI100Classification\PI100Dataset\PI100_test\Product_query\\'  # 测试集地址

image_test = []
labels = []
test_Y = []

dirs = os.listdir(test_dir)
labels = sorted(dirs)  # 类别名

# 加载测试集图片
for label in labels:
    dirname = test_dir + label + '\\'
    for imgfile in os.listdir(dirname):
        if(imgfile[0] == '.'):
            pass
        else:
            image = load_img(dirname + imgfile, target_size=(100, 100, 3))
            image_arr = img_to_array(image)
            image_test.append(image_arr)
            test_Y.append(label)

print('测试集加载完成.....')

le = LabelEncoder()
test_Y = le.fit_transform(test_Y)

model = Sequential()

model = load_model(
    'D:\PI100Classification\PI100Classification\models\origin_model_new_new_next.h5')
model.load_weights(
    'D:\PI100Classification\PI100Classification\models\origin_model_weights_new_new_next.h5')

image_test = np.array(image_test, dtype="float") / 255.0
# image_test = np.array(image_test, dtype="float")

result = model.predict(image_test)
print(result)
print(result.shape)

# 获取每张图片的最大预测值的那个类，并生成
submission = pd.DataFrame(result, columns=le.classes_)
# 输出2000张图片的预测值的表格
submission.to_csv(
    'D:\PI100Classification\PI100Classification\\results\PI100_classification_test_prediction_results.csv'
)
submission = submission.T
FinalSu = submission.idxmax()
FinalSu = pd.Series(FinalSu.reshape(-1, ))

FinalSu.columns = ['图片编号', '分类结果']

# 保存模型预测分类结果
FinalSu.to_csv(
    'D:\PI100Classification\PI100Classification\\results\PI100_classification_test_Final_results.csv'
)


FinalSu = submission.idxmax().reshape(-1, )
pre_Y = le.fit_transform(FinalSu)
print(pre_Y)

# 使用不同标准进行模型评估
PI100_mse = mean_squared_error(test_Y, pre_Y)
PI100_f1_score = f1_score(test_Y, pre_Y, average='micro')  # f1_score
PI100_accuracy_score = accuracy_score(test_Y, pre_Y)  # accuracy_score
PI100_recall_score = recall_score(
    test_Y, pre_Y, average='micro')  # recall_score

print('f1_score:{0}'.format(PI100_f1_score))
print('accuracy_score:{0}'.format(PI100_accuracy_score))
print('recall_score:{0}'.format(PI100_recall_score))
print('MSE:{0}'.format(PI100_mse))
