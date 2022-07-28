# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import re
import detectColor
import time
#print(time.strftime('%m%d', time.localtime(time.time())))
#print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
train_top_dir_file = base_path + '/train/train_top_dir_file.txt'
train_top_label_file = base_path + '/train/train_top_label_file.txt'


val_path = base_path + '/validation'
val_label_dir = base_path + '/validation/annos'
val_top_dir_file = base_path + '/validation/val_dir_file.txt'
val_top_label_file = base_path + '/validation/val_label_file.txt'

new_img_dir = base_path + '/train/image_new/'
new_val_img_dir = base_path + '/validation/image_new/'

def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

old_img_file = ReadFile(train_top_dir_file)
old_img_label_file = ReadFile(train_top_label_file)
version =''
train_x = ReadFile(base_path+'/train/train_x_file_v2.txt')
train_y = ReadFile(base_path+'/train/train_y_file_v2.txt')
new_train_y_file = open(base_path + '/train/train_y_file_v2_1.txt',"w")
new_train_y = []
#len(old_img_label_file)-53918
for i in range(len(train_x)):
    #i = i + 53918 -1

    index = re.sub('.jpg','',train_x[i])
    try:
        index = re.sub('_','',index)
        try:
            index = re.sub('Up','',index)
            index = int(index)
        except:
            index = re.sub('Down','',index)
            index = int(index)
    except:
        index = int(index)

    old_label = int(old_img_label_file[index-1])
    train_y[i] = int(train_y[i])
    if train_y[i] != old_label:
        print(index, train_y[i], old_label)
        break
        train_y[i] = old_label
        #print(index, train_y[i], old_label)
    #new_train_y.append(old_label)
    #new_train_y_file.write(str(new_train_y[i]) + '\n')