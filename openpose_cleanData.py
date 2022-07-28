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

#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'
train_top_dir_file = base_path + '/train/train_top_dir_file.txt'
train_top_label_file = base_path + '/train/train_top_label_file.txt'
val_label_dir = base_path + '/validation/annos'
val_top_dir_file = base_path + '/validation/val_dir_file.txt'
val_top_label_file = base_path + '/validation/val_label_file.txt'

save_path = val_path + '/img_clean/' 
save_path_whole = val_path + '/img_clean_whole/' 
save_path_up = val_path + '/img_clean_up/' 
save_path_down = val_path + '/img_clean_down/' 
#other_path = base_path + '/train/img_del/' 
def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

version ='0505'

# old_img_file = ReadFile(train_top_dir_file)
# old_img_label_file = ReadFile(train_top_label_file)
# train_file_clean = open(base_path + '/train/train_file_clean_'+version+'.txt',"a")

old_val_img_file = ReadFile(val_top_dir_file)
old_val_img_label_file = ReadFile(val_top_label_file)
val_file_clean = open(val_path + '/val_file_clean_'+version+'.txt',"a")

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/irene/local/src/openpose/models/"
    params["face"] = True
    params["hand"] = True

    # -----計算照片上下左右界線
    def count_limitPoint(space, percent, point1, point2, max, min):
        #判斷點往旁邊取
        limitpoint1 = point1 - space*percent
        limitpoint2 = point2 + space*percent
        #print("origin point 1 & 2 :",limitpoint1,limitpoint2)
        #print(" space*percent", space*percent)
        if limitpoint1 < min:
            limitpoint1 = min
        if limitpoint2 >max :
            limitpoint2 = max
        if limitpoint1 > limitpoint2:
            limitpoint1 = min
            limitpoint2 = max
        #print("point 1 & 2 , min max",limitpoint1,limitpoint2,min,max)
        return int(limitpoint1), int(limitpoint2)
        
    def count_center(point1,point2):
        point_halfdis = (point2 - point1)/2
        point_center = point1 + point_halfdis
        return int(point_halfdis), int(point_center)

    def openpose_preprocess(img_path,img_label, img_name, new_x_file):
        #bone_point = np.zeros((,3)) #neck, r_shouder, r_elbow, r_wrist, l_shouder, l_elbow, l_wrist, butt, r
        print(img_path)
        if  img_label !='10' and img_label !='11' and img_label !='12' and img_label !='13': #先過濾掉連身類標籤
            #  --------openpose前置作業---------------------------------------------------- 
            # Flags
            parser = argparse.ArgumentParser()
            parser.add_argument("--image_path", default=img_path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
            args = parser.parse_known_args()
            # Add others in path
            for i in range(0, len(args[1])):
                curr_item = args[1][i]
                if i != len(args[1])-1: next_item = args[1][i+1]
                else: next_item = "1"
                if "--" in curr_item and "--" in next_item:
                    key = curr_item.replace('-','')
                    if key not in params:  params[key] = "1"
                elif "--" in curr_item and "--" not in next_item:
                    key = curr_item.replace('-','')
                    if key not in params: params[key] = next_item
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            # Process Image
            datum = op.Datum()
            img = cv2.imread(args[0].image_path)
            if img is None:
                print("img not exist!")
                return 0
            datum.cvInputData = img
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            height, width, channels = img.shape
            #骨架圖
            #cv2.imwrite(save_path + img_name+'bone_Down.jpg', datum.cvOutputData)
            #  ------------------------------------------------------------------------------       

            #  --------根據骨架的圖片切割---------------------------------------------------- 
            #判斷圖片有人
            a = np.zeros(3)
            if datum.poseKeypoints is not None: 
                #去除沒人或多人照
                try:
                    if (datum.poseKeypoints[1][1] == a).any() == False:
                        print("!!except >=2 human")
                        return 0
                    if (datum.poseKeypoints[0]== a).any():
                        print("!!NO human")
                        return 0
                except:
                    pass
                #去除大頭照
                if (datum.poseKeypoints[0][0] == a).any() == False:
                        try:
                            if ((datum.poseKeypoints[0][1][1] - datum.poseKeypoints[0][0][1]) >= (width/4)):
                                print("!!except Big head photo")
                                return 0
                        except Exception as e:
                            print("!!except Big head photo",e)

                #label為下身
                if(int(img_label) > 6): 
                    try:
                        #判斷正常站立
                        if datum.poseKeypoints[0][12][0] < datum.poseKeypoints[0][9][0]:
                            print("!!not front phto")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0
                        if datum.poseKeypoints[0][13][1] < datum.poseKeypoints[0][12][1]:
                            print("!!knee > butt")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0
                        if datum.poseKeypoints[0][14][1] < datum.poseKeypoints[0][13][1]:
                            print("!!ankle > knee")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0
                        if datum.poseKeypoints[0][13][0] < datum.poseKeypoints[0][10][0] or  datum.poseKeypoints[0][14][0] < datum.poseKeypoints[0][11][0]:
                            print("!!ankle > knee")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0
                    except Exception as e:
                        print("!!normal down excepttion",e)

                    #偵測到全身
                    if (datum.poseKeypoints[0][10] == a).any() == False :
                        print("Have upper & lower")
                        cv2.imwrite(save_path_whole + img_name, img)
                    else:
                        cv2.imwrite(save_path_down + img_name, img)
                    cv2.imwrite(save_path + img_name, img)
                    new_x_file.write(str(img_name)+'#'+img_label+'\n')
                #label為上身
                else: 
                    #判斷正常上身姿勢
                    try:
                        if datum.poseKeypoints[0][5][0] < datum.poseKeypoints[0][2][0]:
                            print("!!not front phto")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0
            
                        if datum.poseKeypoints[0][4][1] < datum.poseKeypoints[0][3][1] and datum.poseKeypoints[0][7][1] < datum.poseKeypoints[0][6][1]:
                            print("!!both hands are bending")
                            #cv2.imwrite(other_path+ img_name, img)
                            return 0

                    except Exception as e:
                        print("!!normal upper excepttion",e)
                        return 0

                    #偵測到全身
                    if (datum.poseKeypoints[0][10] == a).any() == False :
                        print("Have upper & lower")
                        cv2.imwrite(save_path_whole + img_name, img) 
                    else:
                        cv2.imwrite(save_path_up + img_name, img)
                    
                    cv2.imwrite(save_path + img_name, img) 
                    new_x_file.write(str(img_name)+'#'+img_label+'\n')  


    for i in range(len(old_val_img_file)):
        #i = i+ 47730
        #openpose_preprocess(train_path+'/'+old_img_file[i], old_img_label_file[i],re.sub('image/','',old_img_file[i]), train_file_clean)   
        openpose_preprocess(val_path+'/'+old_val_img_file[i], old_val_img_label_file[i],re.sub('image/','',old_val_img_file[i]), val_file_clean)   


    #train_file_clean.close()
    val_file_clean.close()
except Exception as e:
        print(e)
        sys.exit(-1)
