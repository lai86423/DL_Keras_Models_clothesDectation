#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt ?�於顯示?��?
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
import pandas as pd
from keras.models import Model
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'
pltsave_path = base_path+'/plt_img'
error_img_path = base_path+ '/errorPredict_img/pattern/'
# y_true = [1, 1, 2, 4, 3, 5, 3, 3, 5,4,1,2]
# y_pred = [0, 1, 2, 2, 0, 4, 3, 4, 5,4,2,1]
def printFun(a):
    for i in range(20):
        print(a[i])

def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data
def predectTest(name, date):
    #Load Data
    #x_val = np.load(os.path.join(train_path,'inputs1val_sleeve_0426nobg.npy'))
    #y_val = np.load(os.path.join(train_path,'labels1val_sleeve_0426nobg.npy'))

    x_data = np.load(os.path.join(train_path,'inputs1train_pattern.npy'))
    y_data = np.load(os.path.join(train_path,'labels1train_pattern.npy'))
    print(len(x_data),len(y_data))
    datasize = int(2861*0.2)
    x_val = x_data[-datasize:]
    y_val = y_data[-datasize:]
    print(len(x_val),len(y_val))


    #x_data = ReadFile(train_path+'/trainval_sleeve_0426nobg.txt')
    #error_img_data = open(error_img_path+'/pattern0524.txt','w')
    y_val = np.round(y_val,4)
    print('y_val shape', y_val.shape)
    printFun(y_val)
    y_true = np.zeros((len(y_val),1))

    #取最大值INDEX
    for i in range(len(y_val)):
        y_true[i] = np.argmax(y_val[i])
    printFun(y_true)
    print('y_true shape',name, y_true.shape)

    #  ---------Model 跑預測值
    model = keras.models.load_model(base_path+'/model/EfficientNetL2_pattern_0524.h5')
    y_pred = model.predict(x_val)
    y_pred = np.round(y_pred,4)
    print("y_pred shape = ", y_pred.shape)
    printFun(y_pred)

    #取最大值INDEX
    max_pred = np.zeros((len(y_val),1))
    for i in range(len(y_pred)):
        max_pred[i] = np.argmax(y_pred[i])
        if y_true[i] != max_pred[i]:
            #x_val[i] = re.sub('/home/irene/deepfashion2/DeepFashion2Dataset/train/img_hand_new/','', x_data[i])
            #print(x_val[i])
            #error_img_data.write(str(y_true[i])+str(max_pred[i])+'\n')
            cv2.imwrite(error_img_path+str(i)+ str(y_true[i])+str(max_pred[i])+'.jpg', x_val[i])

    print("after arg y_pred = ", max_pred.shape)
    printFun(max_pred)
    sns.set(font_scale=0.4)

    confusion = confusion_matrix(y_true, max_pred)
    print("----------")
    print('Confusion Matrix\n')
    print(confusion,type(confusion))
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='PuBuGn')
    plt.title(name + date)
    plt.savefig(pltsave_path +'/'+name + date +'.png') 
    plt.show()

    output_file = open(pltsave_path+'/'+ name + date +'1.txt', 'w') 
    output_file.write(str(confusion)+'\n')

    new_confu = np.zeros((confusion.shape))
    #print("new_confu", new_confu)
    output_file2 = open(pltsave_path+'/'+ name + date +'2.txt', 'w') 
    for i in range(len(confusion)):
        list_sum = np.sum(confusion[i])
        #print(confusion[i], list_sum, type(confusion[i]))
        for j in range(len(confusion[i])):
            new_confu[i][j] = confusion[i][j]/list_sum
        output_file2.write(str(new_confu[i])+'\n')
        #print("--", new_confu[i])
    print(new_confu)
    

predectTest('df2_pattern','0524')