
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'
img_path = base_path + '/train/img_body/' 

path = 'cloth_patteren.csv'
with open(path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for filename in os.listdir(img_path):
        writer.writerow([filename])
