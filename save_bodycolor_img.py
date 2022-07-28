#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np 
import os
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset/train'
img_dir = base_path + '/img_body/'
new_img_dir = base_path + '/img_body_coat/'
x_data_path = base_path + '/traintrain_coat_0519_nohand.txt'
def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

x_data = ReadFile(x_data_path)

for i in range(len(x_data)):
    try:
        frame = cv2.imread(x_data[i])
        cv2.imwrite(new_img_dir+ str(i) +'.jpg', frame)
    except:
        pass