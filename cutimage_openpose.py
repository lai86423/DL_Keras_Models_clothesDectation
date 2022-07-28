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
new_img_bone_dir = base_path + '/train/bone_line/'
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

    # def mask_line(point1, point2, point3,image,thickness, save_path, img_name):
    #     # Convert uint8 to float
    #     foreground = image.astype(float)

    #     mask = np.full(image.shape,(0,0,0), np.uint8)
    #     # mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    #     cv2.line(mask, (point1[0],point1[1]), (point2[0],point2[1]), (255, 255, 255), thickness=thickness)
    #     #alpha = cv2.bitwise_and(image, mask)
    #     cv2.imwrite(save_path + img_name+'mask.jpg', mask)
    #     # Normalize the alpha mask to keep intensity between 0 and 1
    #     alpha = mask.astype(float)/255
    #     cv2.imwrite(save_path + img_name+'alpha.jpg', alpha)
    #     # Multiply the foreground with the alpha matte
    #     foreground = cv2.multiply(alpha, foreground)

    #     # Multiply the background with ( 1 - alpha )
    #     #background = cv2.multiply(1.0 - alpha, background)

    #     # Add the masked foreground and background.
    #     #outImage = cv2.add(foreground, background)

    #     # Display image
    #     cv2.imwrite(save_path + img_name+'test.jpg', foreground/255)
    #     cv2.waitKey(0)


    # def mask_line(point1, point2, point3,image,thickness, save_path, img_name):
    #     print("mask")
    #     mask = np.full(image.shape,(0,0,0), np.uint8)
    #     cv2.line(mask, (point1[0],point1[1]), (point2[0],point2[1]), (255, 255, 255), thickness=thickness)
        
    #     bitwiseAnd = cv2.bitwise_and(image, mask)
    #     print(image.shape, mask.shape )

    #     final = bitwiseAnd[int(min(point1[1],point2[1])):int(max(point1[1],point2[1])),int(min(point1[0],point2[0])):int(max(point1[0],point2[0]))]
        
    #     # # -----顏色辨識
    #     color = detectColor.get_color(final,img_name+'line')
    #     #color2 = detectColor.get_color(mask2,img_name+'line2')
    #     print("## Color = ", color)
    #     cv2.imwrite(save_path + img_name+ color+'.jpg', final)
    #     #cv2.imshow("AND", bitwiseAnd)
    #     cv2.waitKey(0)
    def mask_line2(point1, point2, point3,image,thickness, save_path, img_name):
        print("mask")
        mask1 = np.full(image.shape,(0,0,0), np.uint8)
        cv2.line(mask1, (point1[0],point1[1]), (point2[0],point2[1]), (255, 255, 255), thickness=thickness)
        mask1 = cv2.bitwise_and(image, mask1) #shape (1320, 880, 3)
        mask2 = np.full(image.shape,(0,0,0), np.uint8)
        cv2.line(mask2, (point3[0],point3[1]), (point2[0],point2[1]), (255, 255, 255), thickness=thickness)
        mask2 = cv2.bitwise_and(image, mask2) #shape (1320, 880, 3)

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

        print(BGR_max, BGR_max2)
        final = cv2.bitwise_and(mask1, mask2)
        cv2.imwrite(save_path + img_name+ 'AND.jpg', final)
        #cv2.imshow("AND", bitwiseAnd)
        cv2.waitKey(0)

    def openpose_preprocess(img_path,img_label, img_name, save_path, new_x_file, new_y_file):
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
                        print("Have upper & lower")
                        mask_line2(datum.poseKeypoints[0][2], datum.poseKeypoints[0][3], datum.poseKeypoints[0][4], img, 40, save_path, img_name)

                        #---切開上身
                        # 1.切上下
                        try:
                            legH = datum.poseKeypoints[0][13][1] - datum.poseKeypoints[0][12][1]
                            if legH <=0:
                                print("!!except legH")
                                return 0
                        except Exception as e:
                            print("!!except legH",e)
                            return 0  
            #             try:
            #                 img_Up_height_N,  img_Up_height_S = count_limitPoint(legH, 0.1, int(datum.poseKeypoints[0][1][1]), int(datum.poseKeypoints[0][8][1]),height, 0 )
            #             except Exception as e:
            #                 print("!!except img_Up_height_N,  img_Up_height_S except",e)
            #                 return 0
                            

            #             # 2.切左右
            #             try:
            #                 shoWid = datum.poseKeypoints[0][5][0] - datum.poseKeypoints[0][2][0]
            #                 if shoWid <=0:
            #                     print("!!except shoWid")
            #                     return 0
            #             except Exception as e:
            #                 print("!!except shoWid",e)
            #                 return 0
            #             try:
            #                 img_Up_width_L,  img_Up_width_R= count_limitPoint(shoWid, 0.3, int(datum.poseKeypoints[0][3][0]),int(datum.poseKeypoints[0][6][0]),width, 0)

            #             except Exception as e:
            #                 print("!!except img_Up_width_L,  img_Up_width_R except",e)
            #                 return 0
                        
            #             # 3.切圖
            #             try:
            #                 img_Up = img[img_Up_height_N:img_Up_height_S,img_Up_width_L:img_Up_width_R]
            #                 #print("img_Up OK!")
            #             except Exception as e:
            #                 print("!!except img_Up ",e)
            #                 return 0

            #             # 判斷此切圖是否為原本標籤內容
            #             if int(img_label) <= 5:
            #                 if (img_Up.shape== a).any(): 
            #                     print("photo size too small")
            #                     return 0
            #                 cv2.imwrite(save_path + img_name+'_Up.jpg', img_Up)
            #                 new_x_file.write(str(img_name)+'_Up.jpg' + '\n')
            #                 #new_y_file.write(img_label + '\n')

            #                 # # -----顏色辨識
            #                 # height_end, width_end, channels = img_Up.shape
            #                 # try:
            #                 #     img_Color = img_Up[int(width_end/4) :int(width_end*3/4), int(height_end/4):int(height_end*3/4)]
            #                 #     print("img_Color OK!",img_Color.shape)

            #                 # except Exception as e:
            #                 #     print("!!except img_Color img_Up")
            #                 #     return 0

            #                 # if (img_Color.shape== a).any(): 
            #                 #     print("color photo size too small")    
            #                 #     return 0                       
            #                 # cv2.imwrite(save_path+'color/' + img_name+'_color.jpg', img_Color)
            #                 # color = detectColor.get_color(img_Color,img_name)
            #                 # print("## Color = ", color)
            #             else:
            #                 #不是的部分還沒有標籤 另外存
            #                 if (img_Up.shape== a).any(): 
            #                     print("photo size too small")
            #                     return 0
            #                 cv2.imwrite(save_path +'img_nolabel/'+ img_name+'_Up.jpg', img_Up)
                            
            #             #---切開下身
            #             # 1.切上下
            #             try:
            #                 img_Down_height_N,  img_Down_height_S = count_limitPoint(legH, 1, int(datum.poseKeypoints[0][10][1]), int(datum.poseKeypoints[0][10][1]),height, int(datum.poseKeypoints[0][8][1])-0.1*legH )
            #             except Exception as e:
            #                 print("!!except img_Down_height_N,  img_Down_height_S ",e)
            #                 return 0
            #             # 2.切左右
            #             try:
            #                 buttWid = datum.poseKeypoints[0][12][0] - datum.poseKeypoints[0][9][0]
            #                 if buttWid <=0:
            #                     print("!!except buttWid")
            #                     return 0
                            
            #             except Exception as e:
            #                 print("!!except buttWid",e)
            #                 return 0
            #             try:
            #                 img_Down_width_L,  img_Down_width_R= count_limitPoint(buttWid, 1, int(datum.poseKeypoints[0][9][0]),int(datum.poseKeypoints[0][12][0]),width, 0)

            #             except Exception as e:
            #                 print("!!exceptimg_Down_width_L,  img_Down_width_R except",e)
            #                 return 0
                        
            #             # 3.切圖                
            #             try:
            #                 img_Down = img[img_Down_height_N:img_Down_height_S,img_Down_width_L:img_Down_width_R]
            #                 #print("img_Down OK!")
            #             except Exception as e:
            #                 print("!!except img_Down ",e)
            #                 return 0
                        
            #             if int(img_label) > 5:
            #                 if (img_Down.shape== a).any(): 
            #                     print("photo size too small")
            #                     return 0
            #                 cv2.imwrite(save_path + img_name+'_Down.jpg', img_Down)
            #                 new_x_file.write(str(img_name)+'_Down.jpg' + '\n')
            #                 #new_y_file.write(img_label + '\n')
            #                 # # -----顏色辨識
            #                 # height_end, width_end, channels = img_Down.shape
            #                 # img_Color = img_Down[int(width_end/4) :int(width_end*3/4), int(height_end/4):int(height_end*3/4)]
            #                 # if (img_Color.shape== a).any(): 
            #                 #     print("photo size too small")
            #                 #     return 0
            #                 # cv2.imwrite(save_path +'color/' + img_name+'_color.jpg', img_Color)
            #                 # color = detectColor.get_color(img_Color,img_name)
            #                 # print("## Color = ", color)
            #             else:
            #                 if (img_Down.shape== a).any(): 
            #                     print("photo size too small")
            #                     return 0
            #                 cv2.imwrite(save_path +'img_nolabel/'+ img_name+'_Down.jpg', img_Down)
                        

            #         #--只有上身  
            #         else:
            #             print("Only upper")
                        
            #             # 1.切上下
            #             try:
            #                 shoWid = datum.poseKeypoints[0][5][0] - datum.poseKeypoints[0][2][0]
            #                 if shoWid <=0:
            #                     print("!!except shoWid")
            #                     return 0
            #             except Exception as e:
            #                 print("!!except shoWid",e)
            #                 return 0
            #             try:
            #                 img_Up_height_N,  img_Up_height_S = count_limitPoint(shoWid, 0.4, int(datum.poseKeypoints[0][1][1]), int(datum.poseKeypoints[0][8][1]),height, 0 )
            #             except Exception as e:
            #                 print("!!except img_Up_height_N,  img_Up_height_S ",e)
            #                 return 0
            #             # 2.切左右
            #             try:
            #                 img_Up_width_L,  img_Up_width_R= count_limitPoint(shoWid, 0.3, int(datum.poseKeypoints[0][3][0]),int(datum.poseKeypoints[0][6][0]),width, 0)

            #             except Exception as e:
            #                 print("!!except img_Up_width_L,  img_Up_width_R ",e)
            #                 return 0
            #             # 3.切圖
            #             try:
            #                 img_Up = img[img_Up_height_N:img_Up_height_S,img_Up_width_L:img_Up_width_R]
            #             except Exception as e:
            #                 print("!!except img_Up ",e)
            #                 return 0
            #             if (img_Up.shape== a).any(): 
            #                 print("photo size too small")
            #                 return 0
            #             cv2.imwrite(save_path + img_name+'_Up.jpg', img_Up)
            #             new_x_file.write(str(img_name)+'_Up.jpg' + '\n')
            #             #new_y_file.write(img_label + '\n')
            #             # # -----顏色辨識
            #             # height_end, width_end, channels = img_Up.shape
            #             # w_color = int(width_end/3)
            #             # img_Color = img_Up[int(width_end/4) :int(width_end*3/4), int(height_end/4):int(height_end*3/4)]
            #             # if (img_Color.shape== a).any(): 
            #             #     print("photo size too small")
            #             #     return 0
            #             # cv2.imwrite(save_path +'color/' + img_name+'_color.jpg', img_Color)
            #             # color = detectColor.get_color(img_Color,img_name)
            #             # print("## Color = ", color)

            #     #--只有下身 
            #     else:
            #         print("Only lower")
            #         # 1.切上下
            #         try:
            #             legH = datum.poseKeypoints[0][13][1] - datum.poseKeypoints[0][12][1]
            #             if legH <=0:
            #                     print("!!except legH")
            #                     return 0
            #         except Exception as e:
            #             print("!!except legH",e)
            #             return 0
            #         try:
            #             img_Down_height_N,  img_Down_height_S = count_limitPoint(legH, 1, int(datum.poseKeypoints[0][10][1]), int(datum.poseKeypoints[0][10][1]),height, 0 )
            #         except Exception as e:
            #             print("!!except except img_Down_height_N,  img_Down_height_S ",e)
            #             return 0
            #         # 2.切左右
            #         try:
            #             buttWid = datum.poseKeypoints[0][12][0] - datum.poseKeypoints[0][9][0]
            #             if buttWid <=0:
            #                 print("!!except buttWid ")
            #                 return 0
            #         except Exception as e:
            #             print("!!except buttWid ",e)
            #             return 0
            #         try:
            #             img_Down_width_L,  img_Down_width_R= count_limitPoint(buttWid, 1, int(datum.poseKeypoints[0][9][0]),int(datum.poseKeypoints[0][12][0]),width, 0)

            #         except Exception as e:
            #             print("!!except img_Down_width_L,  img_Down_width_R ",e)
            #             return 0
            #         # 3.切圖            
            #         try:
            #             img_Down = img[img_Down_height_N:img_Down_height_S,img_Down_width_L:img_Down_width_R]
            #         except Exception as e:
            #             print("Exception img_Down ",e)
            #             return 0

            #         if int(img_label) > 5:
            #             if (img_Down.shape== a).any(): 
            #                 print("photo size too small")
            #                 return 0
            #             cv2.imwrite(save_path + img_name+'_Down.jpg', img_Down)
            #             new_x_file.write(str(img_name)+'_Down.jpg' + '\n')
            #             #new_y_file.write(img_label + '\n')
            #             # # -----顏色辨識
            #             # height_end, width_end, channels = img_Down.shape
            #             # img_Color = img_Down[int(width_end/4) :int(width_end*3/4), int(height_end/4):int(height_end*3/4)]
            #             # if (img_Color.shape== a).any(): 
            #             #     print("photo size too small")
            #             #     return 0
            #             # cv2.imwrite(save_path+'color/'  + img_name+'_color.jpg', img_Color)
            #             # color = detectColor.get_color(img_Color,img_name)
            #             # print("## Color = ", color)
            #         else:
            #             if (img_Down.shape== a).any(): 
            #                 print("photo size too small")
            #                 return 0
            #             cv2.imwrite(save_path +'img_nolabel/'+ img_name+'_Down.jpg', img_Down)
                    
            # else:
            #     print("No human!")  
            #     cv2.imwrite(save_path + img_name+'.jpg', img) 
            #     new_x_file.write(str(img_name)+'.jpg' + '\n')
                #new_y_file.write(img_label + '\n')
                # # -----顏色辨識
                # img_Color = img[int(width/4) :int(width*3/4), int(height/4):int(height*3/4)]
                # if (img_Color.shape== a).any(): 
                #     print("photo size too small")
                #     return 0
                # cv2.imwrite(save_path +'color/' + img_name+'_color.jpg', img_Color)
                # color = detectColor.get_color(img_Color,img_name)
                # print("## Color = ", color)
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
    for i in range(30):
        i = i+3622
        openpose_preprocess(train_path+'/'+old_img_file[i], old_img_label_file[i],str(i+1).zfill(6), new_img_bone_dir, train_x_file, train_y_file)   


    train_x_file.close()
    train_y_file.close()
    val_x_file.close()
    val_y_file.close()
except Exception as e:
        print(e)
        sys.exit(-1)
