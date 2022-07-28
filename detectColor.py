import  cv2
import numpy as np
import colorList
import os
import remove_bg


#處理圖片
def get_color(frame,name):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color_area = {}
    color = None
    color_dict = colorList.getColorList()
    color_percent = {}
    ddd = {}
    Sum = 0
    for d in color_dict: #輪流計算顏色面積
        #切割出指定顏色區域
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1]) 
        #if d =='skin' or d == 'orange':
            #print(d)
            #cv2.imwrite(path+name+d+'.jpg',mask)
        #影像二值化
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        #影像膨脹
        binary = cv2.dilate(binary,None,iterations=2)
        #輪廓檢測 
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            #輪廓面積
            sum+=cv2.contourArea(c)
        if sum > maxsum :
            #最多面積之顏色 判斷為該顏色
            maxsum = sum
            color = d[:1]
        Sum += sum
        color_percent[d] = sum
    for k,v in  color_percent.items() :
        #print(key)
        v = round(v/Sum, 2)
        ddd[k] = v

    #print("color_percent = ",color_percent)
    

    #彩度
    #hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    Schannel = np.max(hsv[:,1,:])
    Vchannel = np.max(hsv[:,:,1]) #255
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    s = round(np.mean(hsv[:,:,1]),2)
    v = round(np.mean(hsv[:,:,2]),2)
    print(name,s,color)
    print("new color_percent = ",ddd)
    return color, s


def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

if __name__ == '__main__':
    base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
    train_path = base_path + '/train'    
    val_path = base_path + '/validation'    
    img_dir = train_path+'/body_color_clear/'
    #img_dir_S = train_path+'/body_color_clear/thesis/'
    #img_dir_S_result = train_path+'/body_color_clear/thesis/result/'

    img_dir_S = val_path +'/video_colortest/'
    img_dir_S_result = val_path +'/video_colortest/result/'


    new_img_dir = train_path+'/img_body_small/wrong/'
    true_img_dir = train_path+'/img_body_small/true/'
    f_x_dir = img_dir + '/body_color_x.txt'
    f__y_dir = img_dir + '/body_color_y.txt'
    x_list = ReadFile(f_x_dir)
    y_list = ReadFile(f__y_dir)
    #print(len(x_list),len(y_list))

    AllFile = os.listdir(img_dir_S)
    #print(len(AllFile))
    i = 0
    for filename in AllFile:
        #if i == 0:
        frame = cv2.imread(img_dir_S+filename) 
        new_frame = remove_bg.cutimgBg(frame)
        color, s = get_color(new_frame,filename)
        if(s<=255/2):
            cv2.imwrite(img_dir_S_result +'_' + color + str(s) +'_' +filename , new_frame)
        else:
            cv2.imwrite(img_dir_S_result +'#' + color + str(s) +'#' +filename , new_frame)

        #i += 1

    # wrong_time = 0
    # for i in range(len(x_list)):
    #     if x_list[i] in AllFile: 
    #         #if filename.endswith(".png") or filename.endswith(".jpg"):  
    #         filename =  x_list[i]  
    #         #print(filename)
    #         try:
    #             frame = cv2.imread(img_dir+filename) 
    #             new_frame = remove_bg.cutimgBg(frame)
                
    #             #print(y_list[i], get_color(new_frame,filename))
    #             if (y_list[i] != get_color(new_frame,filename)):
    #                 if y_list[i]!= 'P':
    #                     wrong_time += 1
    #                     #cv2.imwrite(new_img_dir + get_color(new_frame,filename) +'_' + y_list[i]+'_' +filename , new_frame)
    #             #else:
    #             #    cv2.imwrite(true_img_dir + get_color(new_frame,filename) +'_' + y_list[i]+'_' +filename , new_frame)
                    
    #         except:
    #             print("no")
    #             pass

    # print("Accracy = ", (len(x_list)-wrong_time)/len(x_list))