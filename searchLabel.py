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
from keras.utils import np_utils

#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'

val_exist_file_x_path = val_path+'/val_x_down_exist.txt'
val_exist_file_y_path = val_path+'/val_y_down_exist.txt'
val_revise_file_x_path = val_path+'/val_x_down_revised.txt'
val_revise_file_y_path = val_path+'/val_y_down_revised.txt'

save_path_body = base_path + '/train/img_body/' 
save_path_hand = base_path + '/train/img_hand/' 
save_path_val_leg_half = val_path + '/img_leg_half/' 

train_x_file_dir = train_path + '/0420_train_x.txt'
train_y_file_dir = train_path + '/0420_train_y.txt'

# train_x_sleeve = open(train_path+'/0420_train_x_sleeve.txt',"w")
# train_y_sleeve = open(train_path+'/0420_train_y_sleeve.txt',"w")
# train_x_down = open(train_path+'/x_down.txt',"w")
# train_y_down = open(train_path+'/y_down.txt',"w")

pattern_path ='/home/irene/deepfashion2/DeepFashion2Dataset/img_pattern'
pattern_x_dir = train_path + '/pattern_x.txt'
pattern_y_dir = train_path + '/pattern_y.txt'

def img_pattern_file():
    for filename in os.listdir(pattern_path):
        print(filename)
def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

def ReadFile_Label(data_path):
    data = []
    label = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line[:-1]
            s = line.split('#')
            data.append(s[0])
            label.append(re.sub('#','',s[1]))
            
    print(len(data),len(label)) 
    return data,label

def SearchImginDoc(img_path,data, label,train_x_file,train_y_file):
    for filename in os.listdir(img_path):
        #filename = re.sub('.jpg','',filename) + '.jpg'
        #print(filename)
        if filename in data:
            x_after_data = []
            y_after_data = []
            train_x_file.write(str(filename)+'\n')
            train_y_file.write(label[data.index(str(filename))]+'\n')
            #print(data.index(filename))

def RiviseIndex_Sleeve():
    x_data = ReadFile(train_x_file_dir)
    y_data = ReadFile(train_y_file_dir)
    x_after_data = []
    y_after_data = []
    for i in range(len(x_data)):
        if int(y_data[i])<=6 :#and int(y_datga[i])< 10: #'y_data[i] !='10' and y_data[i] !='11' and y_data[i] !='12' and y_data[i] !='13'
            x_after_data.append(x_data[i]) 
            if int(y_data[i])<=2 :
                y_after_data.append(y_data[i]) 
            elif y_data[i]=='3':
                y_after_data.append('1') 
            elif y_data[i]=='4':
                y_after_data.append('2')
            elif y_data[i]=='5' or y_data[i]=='6':
                y_after_data.append('3')
            else:
                y_after_data.append('0')

    for i in range(len(x_after_data)):
        train_x_sleeve.write(x_after_data[i]+'\n')
        train_y_sleeve.write(y_after_data[i]+'\n')

def RiviseIndex_Down(x_file_dir, y_file_dir, new_x_file, new_y_file):
    x_data = ReadFile(x_file_dir)
    y_data = ReadFile(y_file_dir)
    x_after_data = []
    y_after_data = []
    for i in range(len(x_data)):
        if int(y_data[i])>6 :#and int(y_datga[i])< 10: #'y_data[i] !='10' and y_data[i] !='11' and y_data[i] !='12' and y_data[i] !='13'
            x_after_data.append(x_data[i]) 
            if y_data[i]=='7' :
                y_after_data.append('1') #短褲
            elif y_data[i]=='8':
                y_after_data.append('2') #長褲
            elif y_data[i]=='9':
                y_after_data.append('3') #裙子
            else:
                y_after_data.append('0')

    for i in range(len(x_after_data)):
        new_x_file.write(x_after_data[i]+'\n')
        new_y_file.write(y_after_data[i]+'\n')


def RiviseIndex_Coat(x_file_dir, y_file_dir, new_x_file, new_y_file):
    x_data = ReadFile(x_file_dir)
    y_data = ReadFile(y_file_dir)
    x_after_data = []
    y_after_data = []
    for i in range(len(x_data)):

        if int(y_data[i])<=6 :#and int(y_datga[i])< 10: #'y_data[i] !='10' and y_data[i] !='11' and y_data[i] !='12' and y_data[i] !='13'
            x_after_data.append(x_data[i]) 
            if y_data[i]=='3'or y_data[i]=='4':
                y_after_data.append('1') #外套
            else:
                y_after_data.append('2') #上衣

    for i in range(len(x_after_data)):
        new_x_file.write(x_after_data[i]+'\n')
        new_y_file.write(y_after_data[i]+'\n')

def RiviseIndex_Coat_long(x_file_dir, y_file_dir, new_x_file, new_y_file):
    x_data = ReadFile(x_file_dir)
    y_data = ReadFile(y_file_dir)
    x_after_data = []
    y_after_data = []
    for i in range(len(x_data)):
            if y_data[i]=='2':
                x_after_data.append(x_data[i]) 
                y_after_data.append('0') #長袖上衣
            elif y_data[i]=='4':
                x_after_data.append(x_data[i]) 
                y_after_data.append('1') #外套

    for i in range(len(x_after_data)):
        new_x_file.write(x_after_data[i]+'\n')
        new_y_file.write(y_after_data[i]+'\n')

if __name__ == '__main__':
    # origin_val_file = val_path + '/val_file_clean_0505.txt'
    # data, label =ReadFile_Label(origin_val_file) 
    # val_x_down = open(val_exist_file_x_path,"w")
    # val_y_down = open(val_exist_file_y_path',"w")
    # x_file = val_x_down
    # y_file = val_y_down
    # SearchImginDoc(save_path_val_leg_half, data, label, x_file, y_file)
    # val_x_down_revise = open(val_revise_file_x_path,"w")
    # val_y_down_revise = open(val_revise_file_y_path,"w")
    # RiviseIndex_Down(val_exist_file_x_path,val_exist_file_y_path, val_x_down_revise,val_y_down_revise)
    
    train_x_origin = train_path + '/0420_train_x.txt'
    train_y_origin = train_path + '/0420_train_y.txt'
    train_x_coat_revise = open(train_path+'/train_x_coat_revise_long.txt',"w")
    train_y_coat_revise = open(train_path+'/train_y_coat_revise_long.txt',"w")
    RiviseIndex_Coat_long(train_x_origin,train_y_origin, train_x_coat_revise, train_y_coat_revise)
    
