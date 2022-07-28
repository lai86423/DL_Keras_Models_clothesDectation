import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import re


base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'


pattern_path ='/home/irene/deepfashion2/DeepFashion2Dataset/train/img_pattern/'
pattern_x_dir = train_path + '/pattern_x2.txt'
pattern_y_dir = train_path + '/pattern_y2.txt'
pattern_x_file = open(pattern_x_dir,'w')
pattern_y_file = open(pattern_y_dir,'w')

def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

def pattern(name, label):
    pattern_x_file.write(name+'\n')
    pattern_y_file.write(label+'\n')

def img_pattern_file():
    for files in os.listdir(pattern_path):
        print(files)
        if files == 'text':
            for filename in os.listdir(pattern_path+'text'):
                pattern(filename,'1')
        elif files == 'dotted':
            for filename in os.listdir(pattern_path+'dotted'):
                pattern(filename,'2')
        elif files == 'checkered':
            for filename in os.listdir(pattern_path+'checkered'):
                pattern(filename,'3')
        elif files == 'striped':
            for filename in os.listdir(pattern_path+'striped'):
                pattern(filename,'4')
        elif files == 'pattern':
            for filename in os.listdir(pattern_path+'pattern'):
                pattern(filename,'5')
        elif files == 'solid':
            for filename in os.listdir(pattern_path+'solid'):
                pattern(filename,'6')

                
def random():
    x_data = ReadFile(pattern_x_dir)
    y_data = ReadFile(pattern_y_dir)
    print("---x_data len = ", len(x_data))
    print("---y_data len = ",len(y_data))
    state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(state)
    state = np.random.get_state()
    np.random.shuffle(y_data)              
    pattern_x_file.write(x_data)
    pattern_y_file.write(y_data)

img_pattern_file()
# x_val = np.load(os.path.join(train_path,'inputs1train_pattern.npy'))
# y_val_category = np.load(os.path.join(train_path,'labels1train_pattern.npy'))

# print(len(x_val),len(y_val_category))
# datasize = int(2861*0.8)
# x_val = x_val[:datasize]
# y_val_category = y_val_category[:datasize]
# datasize = int(2861*0.2)
# x_data = x_val[-datasize:]
# y_data = y_val_category[-datasize:]
# print(len(x_val),len(y_val_category),len(x_data),len(y_data))