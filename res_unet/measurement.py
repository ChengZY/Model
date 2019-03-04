# -*- coding: utf-8 -*-
"""
@author: Cheng Zhong Yao
"""

import cv2
from skimage import exposure, img_as_float
import numpy as np
from matplotlib import pyplot as plt
import os



'''
set default parameter
path_original : the path saved the masks
path_mask     : the path saves the corresponding masks
ratio         : map the pixel to nm
thick         : the line draw in the images
'''
path_original = '..//measured//'
path_mask = '..//spilt1//images//'
#ratio = 500/604  ##500 nm = 604  pixel
ratio = 300/360  ##300 nm = 360  pixel
thick = 1

'''
find the images under the dir and subdir
Input: the dir contains the images need measurement
Output: the file list in the dir
'''
def traverseDirByOSWalk(path):  
    path = os.path.expanduser(path) 
    file_list=[]
    for (dirname, subdir, subfile) in os.walk(path):  
        for f in subfile:
            if (f !='.DS_Store'):
                file_list.append(f)
    return file_list


'''
map the image and mask according to the name
Input: the path of image and corresponding mask path
Output: the list map the image name and mask name
'''
def merge_maskfilename(path_original,path_mask):
    original = traverseDirByOSWalk(path_original)
    mask_list = traverseDirByOSWalk(path_mask)
#    original = original[1:]
    #mask_list =m ask_list
    cor_ = []
    for original_file_name in original:
        file = [original_file_name]
        for mask_file in mask_list:
            if (original_file_name.split('.')[0] in mask_file):
                a = mask_file.split('.')[0].split('_')[-1]
                file.append(mask_file)
#                print(int(a))
        cor_.append(file)
    return cor_
#plt.imshow(imgray)
#def merge_
    
'''
caculate the width and the heigth of the target1
Input:
img_org   : numpy array, the orginal images with RGB, 3 channels
imgray    : numpy array, the masks with only black background and white target area
filename  : string, the name of the target images
des_path  : string, the destination path to save the results
color     : tuple , the boundary color dran in the original images
thereshold: float,thereshold to binary the masks
DEL       : bool, whether delete the first target, True or False
ratio     : float, map the pixel to nm

Output:
width  : width of the target
height : height of the target
label  : index of the target
'''
def cal_wid_hei(imgray,img_org, filename, des_path, color, thereshold, DEL, ratio):
    global thick
    a = ratio
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, thresh = cv2.threshold(imgray, thereshold, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if DEL:
        contours = contours[1:]
    width = []
    height = []
    label = []
    left_i = 0
    right_i = 0
    for contour in contours:
#        contour[:,:,1] = contour[:,:,1] - 43
        v2 = contour[:,0,:]
        X_min = min(v2[:,0])
        X_max = max(v2[:,0])
        Y_min = min(v2[:,1])
        Y_max = max(v2[:,1])
        width.append(round(a*(X_max - X_min),2))
        height.append(round(a*(Y_max - Y_min),2))
        if (X_min < 512):
                left_i = left_i + 1
                label.append('left ' + str(left_i).zfill(2))
                v1 = v2[np.where(v2[:,0] > X_max- 5)[0]]
                point1 = np.where(v2[:,0] == X_min)[0]
                point1_loc = v2[point1[len(point1)//2]]
                point2 = np.where(v1[:,1] == max(v1[:,1]))[0]
                point2_loc = v1[point2[len(point2)//2]]
                point3 = np.where(v1[:,1] == min(v1[:,1]))[0]
                point3_loc = v1[point3[len(point3)//2]]
                cv2.putText(img_org, 'L' + str(left_i).zfill(2)+ ' '+ str(round(a*(X_max - X_min)*10,1)) ,(max(v2[:,0]) - 385 ,(min(v2[:,1])+max(v2[:,1]))//2-25), font, 0.8,color,2,cv2.LINE_AA)
        else:
                v1 = v2[np.where(v2[:,0] < X_min + 5)[0]]
                right_i = right_i + 1
                label.append('right ' + str(right_i).zfill(2))
                point1 = np.where(v2[:,0] == X_max)[0]
                point1_loc = v2[point1[len(point1)//2]]                
                point2 = np.where(v1[:,1] == max(v1[:,1]))[0]
                point2_loc = v1[point2[len(point2)//2]]
                point3 = np.where(v1[:,1] == min(v1[:,1]))[0]
                point3_loc = v1[point3[len(point3)//2]]
                cv2.putText(img_org,'R' + str(right_i).zfill(2) + ' ' + str(round(a*(X_max - X_min)*10 ,1))  ,(min(v2[:,0]) + 150 ,(min(v2[:,1])+max(v2[:,1]))//2-25), font, 0.8,color,2,cv2.LINE_AA)
                
        cv2.line(img_org, (point1_loc[0],point1_loc[1] ), ( 
                ((point3_loc[1] - point1_loc[1]) * (point3_loc[0] - point2_loc[0]) // (point3_loc[1] - point2_loc[1]) - point3_loc[0])*(-1)
                , point1_loc[1] ), color, thick)
        cv2.line(img_org, ( point2_loc[0],point2_loc[1]), ( point3_loc[0],point3_loc[1] ), color, thick)
    return width, height, label


'''
caculate the width and the heigth of the target2
Input:
img_org   : numpy array, the orginal images with RGB, 3 channels
imgray    : numpy array, the masks with only black background and white target area
filename  : string, the name of the target images
des_path  : string, the destination path to save the results
color     : tuple , the boundary color dran in the original images
thereshold: float,thereshold to binary the masks
DEL       : bool, whether delete the first target, True or False
ratio     : float, map the pixel to nm

Output:
width  : width of the target
height : height of the target
label  : index of the target
the images with the boundary saved in des_path
'''
def cal_wid_hei2(imgray,img_org, filename, des_path, color, thereshold, DEL, ratio):
    global thick
    a = ratio
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, thresh = cv2.threshold(imgray, thereshold, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if DEL:
        contours = contours[1:]
    width = []
    height = []
    label = []
    left_i = 0
    right_i = 0
    for contour in contours:
#        contour[:,:,1] = contour[:,:,1] - 43
        v2 = contour[:,0,:]
        X_min = min(v2[:,0])
        X_max = max(v2[:,0])
        Y_min = min(v2[:,1])
        Y_max = max(v2[:,1])
        width.append(round(a*(X_max - X_min),2))
        height.append(round(a*(Y_max - Y_min),2))
        if (X_min < 512):
                left_i = left_i + 1
                label.append('left ' + str(left_i).zfill(2))
                v1 = v2[np.where(v2[:,0] > X_max- 5)[0]]
                point1 = np.where(v2[:,0] == X_min)[0]
                point1_loc = v2[point1[len(point1)//2]]
                point2 = np.where(v1[:,1] == max(v1[:,1]))[0]
                point2_loc = v1[point2[len(point2)//2]]
                point3 = np.where(v1[:,1] == min(v1[:,1]))[0]
                point3_loc = v1[point3[len(point3)//2]]
                cv2.putText(img_org, str(round(a*(X_max - X_min)*10,1)) ,(max(v2[:,0]) - 250 ,(min(v2[:,1])+max(v2[:,1]))//2 -25), font, 0.8,color,2,cv2.LINE_AA)
        else:
                v1 = v2[np.where(v2[:,0] < X_min + 5)[0]]
                right_i = right_i + 1
                label.append('right ' + str(right_i).zfill(2))
                point1 = np.where(v2[:,0] == X_max)[0]
                point1_loc = v2[point1[len(point1)//2]]                
                point2 = np.where(v1[:,1] == max(v1[:,1]))[0]
                point2_loc = v1[point2[len(point2)//2]]
                point3 = np.where(v1[:,1] == min(v1[:,1]))[0]
                point3_loc = v1[point3[len(point3)//2]]
                cv2.putText(img_org, str(round(a*(X_max - X_min)*10 ,1))   ,(min(v2[:,0]) + 300 ,(min(v2[:,1])+max(v2[:,1]))//2 -25), font, 0.8,color,2,cv2.LINE_AA)
                
        cv2.drawContours(img_org, contour, -1, (0, 0, 255), 2)
        cv2.line(img_org, (point1_loc[0],point1_loc[1] ), ( 
                ((point3_loc[1] - point1_loc[1]) * (point3_loc[0] - point2_loc[0]) // (point3_loc[1] - point2_loc[1]) - point3_loc[0])*(-1)
                , point1_loc[1] ), color, thick)
    cv2.imwrite(des_path + filename, img_org)
    return width, height, label


'''
merge the four patches to one images
one   two
three four
'''
def merge_pic(img1,img2,img3,img4):
    return np.concatenate((np.concatenate((img1,img2),axis = 1),np.concatenate((img3,img4),axis = 1)),axis = 0)

import pandas as pd

'''
main function:
output: 1. measurement results: saved in '..//gtcsv//', the file names map to the images name
        2. images: '..//draw_predict//', images with the boundary
    
'''
def main():
    sum_ = []
    All_file = []
    cor_ = merge_maskfilename(path_original,path_mask)
    img_org_list = []
    for file in cor_:
        img_org = cv2.imread(path_original + file[0])
    #    imgray = cv2.imread('..//annotation//'+ file[0],0)
    #    imgray = cv2.blur(imgray,(5,5))
        img1 = cv2.imread(path_mask + file[1],0)
        img2 = cv2.imread(path_mask + file[2],0)
        img3 = cv2.imread(path_mask + file[3],0)
        img4 = cv2.imread(path_mask + file[4],0)
        imgray = merge_pic(img1,img2,img3,img4)
        cv2.imwrite('..//annotation//'+ file[0], imgray)
        imgray = cv2.blur(imgray,(5,5))
    #    cv2.imwrite('..//annotation//'+ file[0], imgray)
    
        width, height, label = cal_wid_hei(imgray,img_org,file[0],'..//gt//', (0,0,255), 127, True , ratio)
    #    width, height= np.array(width), np.array(height)
        data = pd.DataFrame({'label':label, 'width': width, 'height': height}).sort_values(by=['label']).set_index('label')
        data2 = pd.DataFrame({'label':['max','min','average','var'], 
                              'width': [data.width.max(),data.width.min(),round(data.width.mean(),2),round(data.width.std(),2)], 
                              'height': [data.height.max(),data.height.min(),round(data.height.mean(),2),round(data.height.std(),2)]}).set_index('label')
        result = pd.concat([data,data2])
        result.to_csv('..//gtcsv//' + file[0].split('.')[0] + '.csv',index = True)
        sum_.append(result)
        All_file.append(file[0])
        img_org_list.append(img_org)
    
    
    path = '..//predicted//'
    file_list_2 = traverseDirByOSWalk(path)
    for file, file_org in zip(file_list_2, img_org_list):
        img_org = file_org
        cv2.imwrite('..//gt//'+ file, img_org)
        img_pred = cv2.imread(path + file, 0)
        img_pred = cv2.blur(img_pred,(3,3))
        width, height, label = cal_wid_hei2(img_pred,img_org, file, '..//draw_predict//', (0,255,0), 127, False, ratio)


'''
input:  binary mask
output: biggest cnt
'''
def get_contour(mask1):
    
    mask1 = cv2.blur(mask1, (2,2))
    cv2.rectangle(mask1,(0,0),(mask1.shape[1],mask1.shape[0]), (0,0,0), 1)
    imgray = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for cnt in contours:
            area = 0
            if area < cv2.contourArea(cnt):
                cnt2 = cnt
    return cnt2

'''
Input:  contour
output: angel
range(-90, 90)
'''
def get_angle(cnt):
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    return angle

'''
input: cnt, scale
output:width,height
'''
def height_weight(contour, a = 1):
    v2 = contour[:,0,:]
    X_min = min(v2[:,0])
    X_max = max(v2[:,0])
    Y_min = min(v2[:,1])
    Y_max = max(v2[:,1])
    width = round(a*(X_max - X_min),2)
    height = round(a*(Y_max - Y_min),2)

    return width, height
#main()