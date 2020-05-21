"""

"""

import os
import cv2
import json
import numpy as np
from math import *
import math
import matplotlib.pyplot as plt
standard_length = 1000

def cal_angle(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    cos_value = np.cross(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    angle = acos(cos_value)/np.pi*180
    print("angle",angle)

def angle(v1, v2):
    # Vector 1 ; Vector 2
    v1 = [0,0,v1[0],v1[1]]
    v2 = [0,0,v2[0],v2[1]]
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = angle1-angle2
    else:
        included_angle = abs(angle1) + abs(angle2)
        #if included_angle > 180:
            #included_angle = 360 - included_angle
    return included_angle


def Put_on_Canvas(img):
    img = cv2.resize(img,(standard_length,standard_length))
    canvas = np.zeros((3*standard_length,3*standard_length,3),dtype=np.uint8)
    canvas[int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),int(3*standard_length/2 - standard_length/2):int(3*standard_length/2 + standard_length/2),:] =  img
    return canvas

def Move_img(img,x,y):
    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(img, T, (len(img[0]), len(img)))
    return img_translation


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def load_pos(json_path):
    # Reading data back
    with open(json_path, 'r') as f:
        data = json.load(f)
    #print(data["shapes"][0]["points"])
    return data["shapes"][0]["points"]

print("Contact Philip Guo (yawenguo@connect.hku.hk) if you have any questions！")

target_path = "./TargetImages/"
result_path = "./Output/"
lists = os.listdir(target_path)
img_paths = []
for file in lists:
    if ".json" not in file:
        img_paths.append(file)

imgs_arr = []
pos_arr = []

reference_img = cv2.imread(target_path+img_paths[0])
raw_size = len(reference_img)
enlarge_scale = 1000/raw_size
reference_img = Put_on_Canvas(reference_img)
reference_pos = load_pos(target_path+img_paths[0][:-4]+".json")
reference_vector = np.array(reference_pos[1])-np.array(reference_pos[0])
reference_img = Move_img(reference_img,-reference_pos[0][0]*enlarge_scale+500,-reference_pos[0][1]*enlarge_scale+500)

final_img_arr = []
final_img_arr.append(reference_img)
for img_path in img_paths[1:]:
    img = cv2.imread(target_path+img_path)
    raw_size = len(img)
    enlarge_scale = 1000 / raw_size
    img = Put_on_Canvas(img)
    pos = load_pos(target_path+img_path[:-4]+".json")
    vector = np.array(pos[1])-np.array(pos[0])
    rotate_angle = angle(reference_vector,vector) # 顺时针
    img = Move_img(img, -pos[0][0] * enlarge_scale + 500,
                             -pos[0][1] * enlarge_scale + 500)
    img = rotate(img,-rotate_angle)
    final_img_arr.append(img)

for h in range(len(reference_img)):
    if np.sum(reference_img[h,:,:])>0:
        init_y = h
        break
for w in range(len(reference_img[0])):
    if np.sum(reference_img[:,w,:])>0:
        init_x = w
        break

#crop image : x = 125-375 375-625 625-875 y = 200-450  550-800

for i in range(len(final_img_arr)):

    img1 = final_img_arr[i][init_y+250:init_y+550,init_x+200:init_x+500,:]
    img2 = final_img_arr[i][init_y+250:init_y+550,init_x+400:init_x+700,:]
    img3 = final_img_arr[i][init_y+250:init_y+550,init_x+500:init_x+800,:]
    img4 = final_img_arr[i][init_y+550:init_y+850,init_x+200:init_x+500,:]
    img5 = final_img_arr[i][init_y+550:init_y+850,init_x+400:init_x+700,:]
    img6 = final_img_arr[i][init_y+550:init_y+850,init_x+500:init_x+800,:]
    img7 = final_img_arr[i][init_y+150:init_y+450,init_x+400:init_x+700,:]
    img8 = final_img_arr[i][init_y+550:init_y+850,init_x+400:init_x+700,:]

    cv2.imwrite(result_path + "1_"+img_paths[i],img1)
    cv2.imwrite(result_path + "2_"+img_paths[i],img2)
    cv2.imwrite(result_path + "3_"+img_paths[i],img3)
    cv2.imwrite(result_path + "4_"+img_paths[i],img4)
    cv2.imwrite(result_path + "5_"+img_paths[i],img5)
    cv2.imwrite(result_path + "6_"+img_paths[i],img6)
    cv2.imwrite(result_path + "7_"+img_paths[i],img7)
    cv2.imwrite(result_path + "8_"+img_paths[i],img8)

print("==============Finished!")