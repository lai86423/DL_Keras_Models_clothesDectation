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
import colorList
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
new_img_bone_dir = base_path + '/train/bone_line/'
new_val_img_dir = base_path + '/validation/image_new/'
diff_L = []
diff_N= [] 
diff_s = []
del_img = []


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
train_x_file = open(base_path + '/train/train_x_file_v2'+version+'.txt',"a")
train_y_file = open(base_path + '/train/train_y_file_v2'+version+'.txt',"a")

old_val_img_file = ReadFile(val_top_dir_file)
old_val_img_label_file = ReadFile(val_top_label_file)
val_x_file = open(base_path + '/validation/val_x_file'+version+'.txt',"a")
val_y_file = open(base_path + '/validation/val_y_file'+version+'.txt',"a")

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
    
    def mask_line2(point1, point2, point3,point4,image,thickness, save_path, img_name):
        #上手臂上半2/4
        mask1 = np.full(image.shape,(0,0,0), np.uint8)
        mid_point = (int((point1[0]+point2[0])/2),int((point1[1]+point2[1])/2))
        mid_point_up = (int((point1[0]+mid_point[0])/2),int((point1[1]+mid_point[1])/2))
        hand_up1 = (int((mid_point_up[0]+point1[0])/2), int((mid_point_up[1]+point1[1])/2))
        hand_up2 = (int((mid_point[0]+mid_point_up[0])/2), int((mid_point[1]+mid_point_up[1])/2)) 
        cv2.line(mask1, hand_up1, hand_up2, (255, 255, 255), thickness=thickness)
        mask1 = cv2.bitwise_and(image, mask1) #shape (1320, 880, 3)
        L = min(hand_up1[0],hand_up2[0])-6
        R = max(hand_up1[0],hand_up2[0])+6
        N = min(hand_up1[1],hand_up2[1])-6
        S = max(hand_up1[1],hand_up2[1])+6 
        #print(L,R,N,S)
        mask1_end = mask1[int(N):int(S),int(L):int(R)]
        #cv2.imwrite(save_path + img_name+ 'mask1.jpg', mask1_end)

        #下手臂上半1/4
        mask2 = np.full(image.shape,(0,0,0), np.uint8)
        mid_point2 = (int((point2[0]+point3[0])/2),int((point2[1]+point3[1])/2)) 
        mid_point2 = (int((point2[0]+mid_point2[0])/2),int((point2[1]+mid_point2[1])/2)) 
        cv2.line(mask2, (point2[0],point2[1]), mid_point2, (255, 255, 255), thickness=thickness)
        mask2 = cv2.bitwise_and(image, mask2) #shape (1320, 880, 3)
        L = min(point2[0],mid_point2[0])-6
        R = max(point2[0],mid_point2[0])+6
        N = min(point2[1],mid_point2[1])-6
        S = max(point2[1],mid_point2[1])+6 
        #print(L,R,N,S)
        mask2_end = mask2[int(N):int(S),int(L):int(R)]
        #cv2.imwrite(save_path + img_name+ 'mask2.jpg', mask2_end)

        #頸部肩膀下半
        mask3 = np.full(image.shape,(0,0,0), np.uint8)
        mid_point3 = (int((point1[0]+point4[0])/2),int((point1[1]+point4[1])/2))
        cv2.line(mask3, (point1[0],point1[1]), mid_point3, (255, 255, 255), thickness=thickness)
        mask3 = cv2.bitwise_and(image, mask3) #shape (1320, 880, 3)
        L = min(point1[0],mid_point3[0])-6
        R = max(point1[0],mid_point3[0])+6
        N = min(point1[1],mid_point3[1])-6
        S = max(point1[1],mid_point3[1])+6 
        #print(L,R,N,S)
        mask3_end = mask3[int(N):int(S),int(L):int(R)]
        color = detectColor.get_color(mask3_end,img_name+'color')
        cv2.imwrite(save_path +'mask3'+ img_name+color+ '.jpg', image)

        BGR_img = mask1.reshape(-1,3)  #shape (1161600, 3)
        BGR_max = []
        for i in  range(3):
            tmp = [0] * 256
            for im in BGR_img[:,i]:
                tmp[im] += 1
            BGR_max.append(tmp[1:].index(max(tmp[1:])) + 1) #算1~256 哪個數量最多
        
        BGR_img2 = mask2.reshape(-1,3) 
        BGR_max2 = []
        for i in  range(3):
            tmp = [0] * 256
            for im in BGR_img2[:,i]:
                tmp[im] += 1
            BGR_max2.append(tmp[1:].index(max(tmp[1:])) + 1) #算1~256 哪個數量最多

        diff = [BGR_max[i] - BGR_max2[i] for i in range(len(BGR_max))]
        diff = [abs(number) for number in diff]
        diff = sum(diff)
        #final = cv2.bitwise_or(mask1, mask2)
        #cv2.imwrite(save_path +'testError'+ img_name, final)
        
        cv2.waitKey(0)
        # if 'L'in img_name:
        #     if diff >=100:
        #         diff_L.append((diff,img_name))
        # elif 'N' in img_name:
        #     if diff >=100:
        #         diff_N.append((diff,img_name+str(BGR_max)+str(BGR_max2)))

        # else:
        #     if diff <100:
        #         diff_s.append((diff,img_name+str(BGR_max)+str(BGR_max2)))
    
        color = detectColor.get_color(mask1_end,img_name+'sleeve')
        color2 = detectColor.get_color(mask2_end,img_name+'sleeve')
        #hsv_img = cv2.cvtColor(mask1_end, cv2.COLOR_BGR2HSV)

        if color2 == 'skin' or color2 == 'orange':
            if color == 'skin' or color == 'orange':
                if 'N'not in img_name:
                    diff_N.append((diff,img_name,color,color2))
            else:
                if 's'not in img_name:
                    diff_s.append((diff,img_name,color,color2))
        else:
            if 'L'not in img_name:
                    diff_L.append((diff,img_name,color,color2))
        #print(color,color2)

    def openpose_preprocess(img_path, img_name, save_path, line_wid):
        #bone_point = np.zeros((,3)) #neck, r_shouder, r_elbow, r_wrist, l_shouder, l_elbow, l_wrist, butt, r
        print(img_path)
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
        
        params["face"] = False
        params["hand"] = False
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
            try:
                if (datum.poseKeypoints[1][1] == a).any() == False:
                    print("!!except >=2 human")
                    return 0
            except:
                pass
                
            #---判斷有上身
            if (datum.poseKeypoints[0][1] == a).any() == False:
                if (datum.poseKeypoints[0][0] == a).any() == False:
                    try:
                        if ((datum.poseKeypoints[0][1][1] - datum.poseKeypoints[0][0][1]) >= (width/4)):
                            print("!!except Big head photo")
                            return 0
                    except Exception as e:
                        print("!!except Big head photo",e)
                        
                #---判斷同時也具有下身
                if (datum.poseKeypoints[0][10] == a).any() == False :
                    #print("Have upper & lower")
                    #去除手彎狀況
                    ab = np.array([datum.poseKeypoints[0][2][0], datum.poseKeypoints[0][2][1]]) - np.array([datum.poseKeypoints[0][3][0], datum.poseKeypoints[0][3][1]])
                    ad = np.array([datum.poseKeypoints[0][4][0], datum.poseKeypoints[0][4][1]]) - np.array([datum.poseKeypoints[0][3][0], datum.poseKeypoints[0][3][1]])
                    hand_dot = np.dot(ab,ad)
                        
                    #if datum.poseKeypoints[0][3][1] < datum.poseKeypoints[0][4][1]: #去除手腕位置>手肘的圖
                    if hand_dot<=0:
                        mask_line2(datum.poseKeypoints[0][2], datum.poseKeypoints[0][3], datum.poseKeypoints[0][4],datum.poseKeypoints[0][1],  img, line_wid, save_path, img_name)
                        
                    else:
                        ab = np.array([datum.poseKeypoints[0][5][0], datum.poseKeypoints[0][5][1]]) - np.array([datum.poseKeypoints[0][6][0], datum.poseKeypoints[0][6][1]])
                        ad = np.array([datum.poseKeypoints[0][7][0], datum.poseKeypoints[0][7][1]]) - np.array([datum.poseKeypoints[0][6][0], datum.poseKeypoints[0][6][1]])
                        hand_dot = np.dot(ab,ad) 
                        if hand_dot<=0:    
                            mask_line2(datum.poseKeypoints[0][5], datum.poseKeypoints[0][6], datum.poseKeypoints[0][7],datum.poseKeypoints[0][1],  img, line_wid, save_path, img_name)
                            
                        else:
                            del_img.append(img_name)   
         
                    
        return 0

    # for i in range(len(old_img_file)-181893):
    #     i = i + 181893
    #     print("--------", old_img_file[i], old_img_label_file[i])
    #     openpose_preprocess(train_path+'/'+old_img_file[i], old_img_label_file[i],str(i+1).zfill(6), new_img_dir, train_x_file, train_y_file)   
    
    # for i in range(len(old_val_img_file)-11143):
    #     i = i + 11143
    #     print("--------", old_val_img_file[i], old_val_img_label_file[i],i)
    #     openpose_preprocess(val_path+'/'+old_val_img_file[i], old_val_img_label_file[i],'val'+str(i+1).zfill(6), new_val_img_dir, val_x_file, val_y_file)   
    #     openpose_preprocess(val_path+'/'+old_val_img_file[i], old_val_img_label_file[i],'val'+str(i+1).zfill(6), new_val_img_dir, val_x_file, val_y_file)   
    allFileList = os.listdir(train_path+'/sleeve_test/')
 
    extra_num = 1000
    total = 0
    
    #line = [5,10,15,20,25,30]
    print(len(allFileList))
    line = [25]
    for j in line :
        #j = j+2
        diff_L = []
        diff_N= [] 
        diff_s = []
        del_img = []
        filelist=['008932s.jpg','009142N.jpg','004582s.jpg', '006795N.jpg','004552s.jpg','005380N.jpg', '007944s.jpg','008931s.jpg'] #, '004584s.jpg', '003015s.jpg', '001757s.jpg', '000186L.jpg', '001429L.jpg', '002178L.jpg','001751L.jpg', '001162N.jpg'
        print(len(filelist))
        #for i in range(len(filelist)):
        #   openpose_preprocess(train_path+'/sleeve_test/'+filelist[i],filelist[i], new_img_bone_dir, j)   
        for i in range(len(allFileList)):
           openpose_preprocess(train_path+'/sleeve_test/'+allFileList[i],allFileList[i], new_img_bone_dir, j)   
        openpose_preprocess(train_path+'/sleeve_test/001007s.jpg','001007s', new_img_bone_dir, j)   

        total = len(diff_L)+len(diff_N)+len(diff_s)
        print(j,"j, diff=",diff_L, diff_N, diff_s, " len(diff_L)=", len(diff_L)," total=",total)
        if total < extra_num:
            extra_num = total
            best_line_wid = j
    print("best_line_wid",best_line_wid, extra_num)  
    print("del_img", del_img, len(del_img))      
    train_x_file.close()
    train_y_file.close()
    val_x_file.close()
    val_y_file.close()
except Exception as e:
        print(e)
        sys.exit(-1)
