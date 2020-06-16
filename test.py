#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:49:13 2020

@author: fn875
"""
import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

#data = open("/fzi/ids/fn875/no_backup/xzc/pipenv_project/Mask_RCNN/AL_test/dataset_generator/new_labels.csv", "r")
#reader = csv.reader(data)

#for item in reader:
  #  print(item[2])
    
# import os

#imglist = os.listdir('train/pics')
#print(imglist[:])
'''
maskArray = []
num_obj = 5
mask = np.zeros([2054,2456,5], dtype = np.uint8)
for index in range(num_obj):
    img = Image.open('train/Masks/test2/image1/mask%s.png' % (index+1))
    imgArray = np.array(img,dtype=np.uint8)
    maskArray.append(imgArray)
mask = np.array(maskArray,dtype=np.uint8, ndmin = 2)
'''

img = Image.open('dataset_generator/mask/image20/mask4.png')
print(np.max(img))
print(img.getpixel((1248,916)))
'''
file = open('mask_index.txt','r')
all_labels = file.readlines()
labels = all_labels[0]
'''
#img = cv2.imread('train/pics/1.png')
'''
img = Image.open('train/Masks/image1_mask.png')
a = img.getpixel((20,30))
print(a)
img = img.convert('L')
b = img.getpixel((20,30))
c = img.getpixel((21,30))
print(c == b)
# print(img)
# mask = np.zeros([240,320,3], dtype = np.uint8)
# print(mask)
#print(np.max(img))
#image = np.array(img)
#print('____---------______')
#print(image)
'''
'''
for i in range(1,11,1):
    img = Image.open('train/Masks/test/image%d_mask.png'% i)
    img = img.convert('L')
    print(img.getpixel((1248,916)))
    
    image = np.array(img)
    height, width = image.shape
    print(height,width)
    mask = np.zeros([height,width], dtype = np.uint8)
    for v in range(height):
        for u in range(width):
            if image[v,u] == 170:
                mask[v,u] = 2
            elif image[v,u] == 255:
                mask[v,u] = 3
    cv2.imwrite('train/Masks/image%d_mask.png'%i,mask)    
print('done')
'''
# matplotlib.image.imsave('train/Masks/test/test.png', mask,cmap = 'gray')            


# img = Image.open('train/Masks/test/cvtest.png')
#mask = np.zeros([240,320], dtype = np.uint8)
#mask[0,0] = 1
#print(mask[0,0])