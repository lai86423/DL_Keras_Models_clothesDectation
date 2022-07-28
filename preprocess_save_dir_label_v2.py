#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import urllib.request
import json
import re

# 資料路徑--------------------------------------------------------------------
dest_dir ='C:\\Users\\Irene\\Documents\\ncu\\論文\\deepfashion2\\DeepFashion2Dataset'
train_label_dir = dest_dir + '\\train\\annos'
train_img_dir = dest_dir + '\\train\\image'
val_label_dir = dest_dir + '\\validation\\annos'
val_img_dir = dest_dir + '\\validation\\image'

train_top_dir_file = dest_dir + '\\train\\train_top_dir_file.txt'
train_top_label_file = dest_dir + '\\train\\train_top_label_file.txt'
val_top_dir_file = dest_dir + '\\validation\\val_dir_file.txt'

#linux-------
dest_dir ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_label_dir = dest_dir + '/train/annos'
train_img_dir = dest_dir + '/train/image'
val_label_dir = dest_dir + '/validation/annos'
val_img_dir = dest_dir + '/validation/image'

train_top_dir_file = dest_dir + '/train/train_top_dir_file.txt'
train_top_label_file = dest_dir + '/train/train_top_label_file.txt'
val_top_dir_file = dest_dir + '/validation/val_dir_file.txt'
val_top_label_file = dest_dir + '/validation/val_label_file.txt'

# # 讀所有圖資料夾與其內所有圖片 並將路徑寫入img_dir.txt檔案 ----------------------------------------------------------
allFileList = os.listdir(train_label_dir)
train_top_dir = []
train_top_label = []
train_top_dir_file = open(train_top_dir_file,"w")
train_top_label_file = open(train_top_label_file,"w")
#len(allFileList)
k = 0
for i in range(len(allFileList)):
    #print(allFileList[i])
    f = open(train_label_dir+'/'+allFileList[i],"r")
    data = json.load(f)

    try:
        imglabel = data['item1']['category_id']
        #print(data['item1']['category_name'])
        train_top_dir.append('img' + '/' + re.sub('.json','',allFileList[i])+'.jpg')
        train_top_label.append(imglabel)
        #print("same??",imglabel, train_top_label[k])
        train_top_dir_file.write(str(train_top_dir[k]) + '\n')
        train_top_label_file.write(str(train_top_label[k]) + '\n')
        k+=1

    except:
        print("No item1 ",i)
        pass

# allFileList = os.listdir(val_label_dir)
# val_top_dir = []
# val_top_label = []
# val_top_dir_file = open(val_top_dir_file,"w")
# val_top_label_file = open(val_top_label_file,"w")
# len(allFileList)
# k = 0
# for i in range(len(allFileList)):
#     print(allFileList[i])
#     f = open(val_label_dir+'/'+allFileList[i],"r")
#     data = json.load(f)

#     try:
#         imglabel = data['item1']['category_id']
#         #print(data['item1']['category_name'])
#         val_top_dir.append('image' + '/' + re.sub('.json','',allFileList[i])+'.jpg')
#         val_top_label.append(imglabel)
#         #print("same??",imglabel, train_top_label[k])
#         val_top_dir_file.write(str(val_top_dir[k]) + '\n')
#         val_top_label_file.write(str(val_top_label[k]) + '\n')
#         k+=1

#     except:
#         print("No item1 ",i)
#         pass