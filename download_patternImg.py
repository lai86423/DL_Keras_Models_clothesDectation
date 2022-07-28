#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import urllib.request
import json
import csv

dest_dir ='/home/irene/deepfashion2/DeepFashion2Dataset/'
img_dir = dest_dir + 'img_pattern/'

data = dest_dir+'dress_patterns.csv'

# 下載圖片 讀圖片URL跟設定檔名--------------------------------------------------------
#Adding information about user agent
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

# 開啟 CSV 檔案
with open(data, newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:
    #print(row[1])
    if row[1] = ''
    filename = img_dir +row[0]+row[1]+'.jpg'
    image_url = row[3]
    print(filename,image_url)
    # calling urlretrieve function to get resource
    try:
        urllib.request.urlretrieve(image_url, filename)
    except:
        pass



# f = open(train_data,"r")
# data =[]
# data = json.load(f)
# for i in range(len(data['images'])):
#     # setting filename and image URL
#     filename = img_dir +'/'+ data['images'][i]['imageId']+'.jpg'
#     image_url = data['images'][i]['url']
#     print(filename,image_url)
#     # calling urlretrieve function to get resource
#     try:
#         urllib.request.urlretrieve(image_url, filename)
#     except:
#         pass

# f.close()

