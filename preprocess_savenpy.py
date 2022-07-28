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
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
import random

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001)

def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'

val_path = base_path + '/validation'
val_label_dir = base_path + '/validation/annos'


# 製�?訓練資�? 標籤&資�???-----------------------------------------------------
img_per_amount = int(2089) #378#21600 #928

def preprocess(x_path, data_path, x_data_path, y_data_path, name, group_num):
    x_data = ReadFile(x_data_path)
    y_data = ReadFile(y_data_path)
    #datasize = int(len(x_data)*0.2)
    #y_data = np.load(y_data_path, allow_pickle=True)

    print("---x_data len = ",name, len(x_data))
    print("---y_data len = ",name, len(y_data))

    # ?�新?��?資�?
    state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(state)
    state = np.random.get_state()
    np.random.shuffle(y_data)
    
    #x_data = x_data[:datasize]
    #y_data = y_data[:datasize]
    # x_data = x_data[-datasize:]
    # y_data = y_data[-datasize:]

    print("---x_data len = ",name, len(x_data))
    print("---y_data len = ",name, len(y_data))

    #設�?input 維度
    dim1 = 128
    dim2 = 128
    val_non_exist = []

    ##---- 設�?資�???---------------------------------------
    # 存�?練�??�x Npy----------------------------------------
    x_after_data = np.zeros((img_per_amount, dim1, dim2, 3))
    k = 0   # 第k筆npy
    file_cot = 0    # ?��?檔�???    
    y_after_data = np.zeros((img_per_amount))
    output_file = open(data_path+'/'+ 'label'+ name +'.txt', 'w')
    x_output_file = open(data_path+'/'+ 'train'+ name +'.txt', 'w')
    non_exist = []
    
    for i in range(len(x_data)):
        x = x_path+x_data[i]
        #print(x)
        if os.path.isfile(x) and x != []:
            #print(x)
            y_after_data[k] = y_data[i]
            output_file.write(str(y_after_data[k])+'\n')
            x_output_file.write(str(x)+'\n')
            img = cv2.imread(x)#讀??                    
            img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
            img = img_to_array(img)
            x_after_data[k] = img
            k += 1
            #else:
            #    print("--data 10111213")
        else:
            non_exist.append(i)
            #print("--Delete Not File--")                    
        
        if k == img_per_amount:
            print("file_count", file_cot+1)
            non_exist = list(set(non_exist))
            print("non_exist",non_exist, len(non_exist))  
            print(f'x training data', x_after_data.shape)
            np.save(os.path.join(data_path,'inputs' + str(file_cot + 1) + name + '.npy'), x_after_data)
            
            print("Before One Hot y_after_data[k-1]",y_after_data[k-1], y_after_data.shape)
            # One Hot Encoding
            y_after_data = np_utils.to_categorical(y_after_data, group_num)
            print("After One Hot y_after_data[k-1]",y_after_data[k-1], y_after_data.shape)
            np.save(os.path.join(data_path,'labels' + str(file_cot + 1) + name + '.npy'), y_after_data)

            k = 0
            file_cot += 1
            y_after_data = np.zeros((img_per_amount))

    print(k)
    output_file.close() 
    x_output_file.close() 


#train_x_file = train_path + '/train_x_coat_revise_long.txt' 
#train_y_file = train_path + '/train_y_coat_revise_long.txt' 

train_x_file = train_path + '/x_down.txt' #'/0420_train_x_sleeve.txt'
train_y_file = train_path + '/y_down.txt' #'/0420_train_y_sleeve.txt'
val_x_file = val_path + '/val_x_down_revised.txt'
val_y_file = val_path + '/val_y_down_revised.txt'

#preprocess(train_path+'/image_new/', train_path, train_x_file, train_y_file, 'train_up_human', 7) 
#preprocess(train_path+'/img_body/', train_path, train_x_file, train_y_file, 'train_coat_0519_nohand', 2)
#preprocess(train_path+'/img_body/', val_path, train_x_file, train_y_file, 'val_coat_0519_nohand', 2)
#preprocess(train_path+'/img_leg_new/', train_path, train_x_file, train_y_file, 'train_down_0602', 4) 
preprocess(val_path+'/img_leg_new/', val_path, val_x_file, val_y_file, 'val_down_0602', 4) 
#preprocess(val_path+'/image_new/', val_path, val_x_file, val_y_file, 'val_down', 4) 

# pattern_x_dir = train_path + '/pattern_x2.txt'
# pattern_y_dir = train_path + '/pattern_y2.txt'

# pattern_path ='/home/irene/deepfashion2/DeepFashion2Dataset/train/img_pattern/'
# preprocess(pattern_path, train_path, pattern_x_dir, pattern_y_dir, 'train_pattern2', 7)  
#preprocess(pattern_path, train_path, pattern_x_dir, pattern_y_dir, 'val_pattern', 6)  


