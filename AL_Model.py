#!/usr/bin/env python
# coding: utf-8

import os
import csv
import sys
import math
import random
import numpy as np
import cv2
import json
import matplotlib
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

# print(tf.__version__)

# Root directory of the project
ROOT_DIR = os.getcwd()

from mrcnn.config import Config
from mrcnn import model as modellib,utils
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15 # background +  shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320 # 128
    IMAGE_MAX_DIM = 384 # 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 *6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 25
    
config = ShapesConfig()
# config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class BlockDataset(utils.Dataset):
    
    # read txt.file to get labels and mask              
    def get_class(self, image_id):
        '''
        info = self.image_info[image_id]
        with open("all_labels.json") as f:
            temp = json.load(f)
            labels = temp[info]['regions'][1]['region_attributes']['label']
        '''
        
        info = self.image_info[image_id]
        index = info['label_index']
        # file = open('mask_index.txt','r')
        # all_labels = file.readlines()
        data = open('dataset_generator/'+info['csv_path']+'_labels.csv','r')
        all_labels = csv.reader(data)
        labels = []
        for item in all_labels:
            if all_labels.line_num == 1:
                continue
            if item[0] == index:
                labels.append(item[1])

        # labelstr = all_labels[int(index)-1] # This is a sting '[...]' !!
        # labels = labelstr[1:-2].split(',') 
        return labels


    # draw_mask
    def draw_mask(self,num_obj,mask,image,image_id,labels_form):
        # The gray value if input mask_image(gray image) are the order of the classes 1, 2, 3...
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i,j))
                    if at_pixel == index + 1:
                        mask [j,i,index] = 1
            
        return mask
    
    def drawMask(self,image_id):
        info = self.image_info[image_id]
        masklist = os.listdir(info['mask_path'])
        # print(imglist)
        num_obj = len(masklist)
        mask = np.zeros([info['height'],info['width'],num_obj], dtype = np.uint8)
        # mask = np.zeros([info['height'],info['width']], dtype = np.uint8)
        for index in range(num_obj):
            img = Image.open(info['mask_path']+ '/mask%s.png' % (index+1))
            # imgArray = np.array(img,dtype=np.uint8)
            # mask = np.concatenate((mask, np.array(img, dtype = np.uint8)[:],axis = 2)
            for i in range(info['width']):
                for j in range(info['height']):
                    if img.getpixel((i,j)) == 1:
                        mask [j,i,index] = 1                    
        
        return mask
       
        
    def load_shapes(self,count,img_folder,mask_folder,imglist,dataset_root_path):
        # Add classes
        '''
        self.add_class("shapes",1,"large")
        self.add_class("shapes",2,"medium")
        self.add_class("shapes",3,"small")
        '''
        # "red_s","red_m","red_l","yellow_s","yellow_m","yellow_l","green_s",
        # "green_m","green_l","blue_s","blue_m","blue_l","orange_s","orange_m","orange_l"
        
        self.add_class("shapes",1,"red_s")
        self.add_class("shapes",2,"red_m")
        self.add_class("shapes",3,"red_l")
        self.add_class("shapes",4,"yellow_s")
        self.add_class("shapes",5,"yellow_m")
        self.add_class("shapes",6,"yellow_l")
        self.add_class("shapes",7,"green_s")
        self.add_class("shapes",8,"green_m")
        self.add_class("shapes",9,"green_l")
        self.add_class("shapes",10,"blue_s")
        self.add_class("shapes",11,"blue_m")
        self.add_class("shapes",12,"blue_l")
        self.add_class("shapes",13,"orange_s")
        self.add_class("shapes",14,"orange_m")
        self.add_class("shapes",15,"orange_l")

        
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            # print(filestr)
            mask_path = mask_folder +"/image%s" % filestr
            label_index = filestr
            path2Img = img_folder+ "/%s.png" % filestr
            csv_path_str = img_folder.split('/')[-1]
            # print(path2Img)
            cv_img = cv2.imread(path2Img)
            # print(cv_img)
            # resize_img = cv2.resize(cv_img,(384,384),interpolation = cv2.INTER_AREA)
            self.add_image("shapes",image_id = i, path = path2Img,csv_path = csv_path_str,
                            width = cv_img.shape[1],height = cv_img.shape[0],mask_path = mask_path, label_index = label_index)
    
            
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = cv2.imread(info['path'])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # resize_img = cv2.resize(img,(384,384),interpolation = cv2.INTER_AREA)
        return img
    

    def load_mask(self,image_id):
        global iter_num
        info = self.image_info[image_id]
        # Mask of instance individual or not????
        # masklist = os.listdir(info['mask_path'])
        # print(imglist)
        # num_obj = len(masklist)
        # img = Image.open(info['mask_path']+ mask1)
        
        # print(info['mask_path'])
        # num_obj = np.max(img)
        # print('num_obj--->',num_obj)

        labels = []
        labels = self.get_class(image_id)
        # print(labels)
        '''
        for i in range(len(labels)):
            if labels[i].find("_l") != -1:
                labels_form.append("large")
            elif labels[i].find("_m") != -1:
                labels_form.append("medium")
            else:
                labels_form.append("small")
        '''
        class_ids = np.array([self.class_names.index(s) for s in labels])
            
        # The number of instances of each picture
        # num_obj = len(class_ids)
        num_obj = np.max(class_ids)
        # print('num_obj--->',num_obj)
        # mask = np.zeros([info['height'],info['width'],num_obj], dtype = np.uint8)
        # mask = self.draw_mask(num_obj,mask,img,image_id,labels_form)
        mask = self.drawMask(image_id)
        
        '''
        occlusion = np.logical_not(mask[:,:,-1]).astype(np.uint8)
        for i in range(num_obj-2,-1,-1):
            mask[:,:,i] *= occlusion
            occlusion = np.logical_and(occlusion,np.logical_not(mask[:,:,i]))
        '''
        # print('-----END')
        
        return mask.astype(np.bool),class_ids.astype(np.int32)

'''
# initialization
dataset_root_path = "dataset_generator/"
img_folder = dataset_root_path + "labeled_images/train90"
mask_folder = dataset_root_path + "mask/train90"
imglist = os.listdir(img_folder)
# print(imglist)
# count = len(imglist)
train_count = 90 
val_count = 90
# print(count)

# traning data and validation data
dataset_train = BlockDataset()
dataset_train.load_shapes(train_count,img_folder,mask_folder,imglist,dataset_root_path)
dataset_train.prepare()
print('train data preparing finished')
# print("dataset_train-->",dataset_train._image_ids)
# print('-----',dataset_train.get_class(dataset_train.image_ids[0]))
'''
val_dataset_root_path = "dataset_generator/"
val_img_folder = val_dataset_root_path + "labeled_images/val"
val_mask_folder = val_dataset_root_path + "mask/val"
val_imglist = os.listdir(val_img_folder)

val_count = 90
dataset_val = BlockDataset()
dataset_val.load_shapes(val_count,val_img_folder,val_mask_folder,val_imglist,val_dataset_root_path)
dataset_val.prepare()
print('val data preparing finished')
# print("dataset_val-->",dataset_val._image_ids)


# create model in training model
model = modellib.MaskRCNN(mode="training",config = config ,model_dir = MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=40, 
            layers='heads')  


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=40, 
            layers="all")
'''

# Create model object in inference mode.
## For inference
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

infer_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=infer_config)

# Load weights trained on MS-COCO
cur_model = os.path.join( MODEL_DIR,'logs/90_model_v1/best_weights.h5')
model.load_weights(cur_model, by_name=True)

APs = []
for image_id in dataset_val.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    print('-------> finished ',image_id)    
print("mAP: ", np.mean(APs))



