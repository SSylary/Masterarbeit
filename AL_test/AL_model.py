#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import random
import numpy as np
import cv2
import json
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

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
    
# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library

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
    NUM_CLASSES = 1 + 3  # background +  shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320 # 128
    IMAGE_MAX_DIM = 384 #128

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
config.display()

class BlockDataset(utils.Dataset):
    # get the instances of th image
    def get_obj_index(self,image):
        n = np.max(image)
        return n
    
    # read json.file to get labels and mask              
    def get_class(self, image_id):
        '''
        info = self.image_info[image_id]
        with open("all_labels.json") as f:
            temp = json.load(f)
            labels = temp[info]['regions'][1]['region_attributes']['label']
        '''
        # info = self.image_info[image_id]
        file = open('mask_index.txt','r')
        all_labels = file.readlines()
        labels = all_labels[int(image_id)]
        return labels
    
    # draw_mask
    def draw_mask(self,num_obj,mask,image,image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i,j))
                    if at_pixel == index + 1:
                        mask[j,i,index] = 1
        return mask
    
    def load_shapes(self,count,img_folder,mask_folder,imglist,dataset_root_path):
        # Add classes
       
        self.add_class("shapes",1,"large")
        self.add_class("shapes",2,"medium")
        self.add_class("shapes",3,"small")
        
        # "red_s","red_m","red_l","yellow_s","yellow_m","yellow_l","green_s",
        # "green_m","green_l","blue_s","blue_m","blue_l","orange_s","orange_m","orange_l"
        '''
        self.add_class("shapes",1,"red_s")
        self.add_class("shapes",2,"res_m")
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
        '''
        
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_folder +"/image%s_mask.png" % filestr
            # yaml_path = dataset_root_path + "labelme_json/"+filestr+"_json/info.yaml"
            path2Img = img_folder+ "/%s.png" % filestr
            cv_img = cv2.imread(path2Img)
            
            self.add_images("shapes",image_id = i, path = path2Img,
                            width = cv_img.shape[1],height = cv_img.shape[0],mask_path = mask_path)
    
    def load_mask(self,image_id):
        global iter_num
        info = self.image_info[image_id]
        labels = self.get_class(image_id)
        count = 1
        img = Image.open(info['mask_path'])
        # num_obj = self.get_obj_index(img)
        num_obj = len(labels)
        mask = np.zeros([info['height'],info['width'],num_obj], dtype = np.uint8)
        mask = self.draw_mask(num_obj,mask,img,image_id)
        occlusion = np.logical_not(mask[:,:,-1]).astype(np.uint8)
        for i in range(count-2,-1,-1):
            mask[:,:,i] *= occlusion
            occlusion = np.logical_and(occlusion,np.logical_not(mask[:,:,i]))
        labels = []
        labels = self.get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("l") != -1:
                labels.form.append("large")
            elif labels[i].find("m") != -1:
                labels.form.append("medium")
            else:
                labels.form.append("small")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask,class_ids.astype(np.int32)
    
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# initialization
dataset_root_path = "train/"
img_folder = dataset_root_path + "pics"
mask_folder = dataset_root_path + "Masks"
imglist = os.listdir(img_folder)
mask_index = np.load(mask_folder+'/mask_index.npy')
mask_index = mask_index.tolist()
count = len(imglist)

# traning data and validation data
dataset_train = BlockDataset()
dataset_train.load_shapes(count,img_folder,mask_folder,imglist,dataset_root_path)
dataset_train.prepare()

dataset_val = BlockDataset()
dataset_val.load_shapes(7,img_folder,mask_folder,imglist,dataset_root_path)
dataset_val.prepare()

# Load and display random samples
'''
'''
image_ids = np.random.choice(dataset_train.image_ids,4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask,class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image,mask,class_ids,dataset_train.class_names)

'''
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
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')  

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")  

'''



    
                
        
        
        
        
