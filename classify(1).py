# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:26:09 2019

@author: Administrator
"""

import datetime
starttime = datetime.datetime.now()

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
import re
import sklearn
from sklearn.svm import SVC

X = []
Y = []
radius = 1;
n_point = radius * 8;
for i in range(0, 4):
    #遍历文件夹，读取图片
    for f in os.listdir("E:\Python365\photo\%s" % i):
        #打开一张图片并灰度化
        Images = cv2.imread("E:\Python365\photo\%s/%s" % (i, f))
        blured = cv2.blur(Images,(5,5)) 
        image=cv2.resize(blured,(256,256),interpolation=cv2.INTER_CUBIC)
        #lbp=skft.local_binary_pattern(Images,n_point,radius,'default');
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0]) 
        X.append(((hist/255).flatten()))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
#切分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=7)



#随机率为100%（保证唯一性）选取其中的30%作为测试集

#支持向量机
X_train = np.array(x_train)	#训练集向量，要转换成numpy.array型向量
Y_train = np.array(y_train)	#训练集标签
X_test = np.array(x_test)	#测试集向量
Y_test = np.array(y_test)	#测试集标签
clf = SVC(gamma =0.09)	#分类面
clf.fit(X_train, Y_train)	#训练分类面，具体参数参考https://www.cnblogs.com/luyaoblog/p/6775342.html
SVC(C=3, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.09, kernel='Sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

print(clf.predict(X_test))	#打印预测结果
print('训练集精度：' + str(100 * clf.score(X_train,Y_train)) + '%')	#打印精度
print('测试集精度：' + str(100 * clf.score(X_test,Y_test)) + '%')







