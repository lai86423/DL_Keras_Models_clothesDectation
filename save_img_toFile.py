#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import urllib.request
import json
import re

#linux-------
dest_dir ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_label_dir = dest_dir + '/train/body_color_clear/'
img_file_dir = dest_dir + '/train/body_color_clear.txt'
# # 讀所有圖資料夾與其內所有圖片 並將路徑寫入img_dir.txt檔案 ----------------------------------------------------------
allFileList = os.listdir(train_label_dir)
train_top_dir = []
train_top_label = []
img_file = open(img_file_dir,"w")
allFileList.sort()
print(len(allFileList))

for i in range(len(allFileList)):
    #img = train_label_dir+ allFileList[i]
    img_file.write(allFileList[i] + '\n')
    # f = open(train_label_dir+'/'+allFileList[i],"r")
    # data = json.load(f)
