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
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# print(tf.__version__)

from mrcnn.config import Config
from mrcnn import model as modellib,utils
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()
# Local path to trained weights file
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "training_data")
# Load initial weights from coco. Only for first time training
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


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
    IMAGES_PER_GPU = 1

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
        # print('====>labels/'+info['csv_path']+'_labels.csv')
        data = open('labels/'+info['csv_path']+'_labels.csv','r')
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

    # generate the N mask of each image
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
        """
        Add all 15 classes
        "red_s","red_m","red_l","yellow_s","yellow_m","yellow_l","green_s",
        "green_m","green_l","blue_s","blue_m","blue_l","orange_s","orange_m","orange_l"
        """
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
            package_id = (int(filestr)-1)//30 + 1
            package_path = "package%s" % package_id
            # print(filestr)
            if mask_folder == 'mask/training_data/':
                mask_path = mask_folder+package_path +"/image%s" % filestr
               # print('====>',mask_path)
                csv_path_str = "training_data/"+package_path
                path_to_img = img_folder+'/'+package_path+ "/%s.png" % filestr
            else:
                mask_path = mask_folder + "/image%s" % filestr
                csv_path_str = img_folder
                path_to_img = img_folder+ "/%s.png" % filestr
            label_index = filestr
            # path_to_img = img_folder+ "/%s.png" % filestr
            # print(path_to_img)
            cv_img = cv2.imread(path_to_img)
            # print(cv_img)
            # resize_img = cv2.resize(cv_img,(384,384),interpolation = cv2.INTER_AREA)
            self.add_image("shapes",image_id=i, path=path_to_img, csv_path=csv_path_str, width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, label_index=label_index)
    
            
    
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
        
        return mask.astype(np.bool), class_ids.astype(np.int32)


class TrainingProcess:

    def set_training_data(self, src_data=None):
        """
        From AL strategy selected images' list to
        """
        # initial training, using package1 as training data
        if src_data == None:
            imglist = os.listdir("training_data/package1")
        else:
        # training with selected data
        # src_data: list
            imglist = []
            with open(src_data) as f:
                for line in f:
                    imglist.append(line.split('/')[-1][5:])
        return imglist


    def train_model(self, train_data, model_infos, dataset_val, cur_model_path = COCO_MODEL_PATH):
        """
        traing model
        train_data: train_imglist
        model_info: [log_dir, weights_name]
        """

        train_dataset_root_path = img_folder = "training_data"
        train_mask_folder = "mask/training_data/"
        train_imglist = train_data
        train_count = len(train_imglist)
        dataset_train = BlockDataset()
        dataset_train.load_shapes(train_count, img_folder, train_mask_folder, train_imglist, train_dataset_root_path)
        dataset_train.prepare()
        print('train data preparing finished')

        # create model in training model
        config = ShapesConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR, model_info=model_infos)

        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last
        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(cur_model_path, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last()[1], by_name=True)

        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads')

        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also
        # pass a regular expression to select which layers to
        # train by name pattern.
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=20,
                    layers="all")


    def mAP_of_model(self, model_info, val_data):
        infer_config = ShapesConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=infer_config, model_info=model_info)

        # Load weights trained on current model
        cur_model_path = os.path.join(model_info[0], model_info[1])
        cur_model = os.path.join(MODEL_DIR, cur_model_path)
        model.load_weights(cur_model, by_name=True)

        APs = []
        for image_id in val_data.image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        mAP = np.mean(APs)
        with open('mAP_of_Model.txt', 'a') as f:
            f.write(cur_model_path, str(np.mean(APs)))
            f.write('\n')


class ActiveLearningStrategy:
    """
    Using different strategies to select most 'hard' images for optimizing the first version model.
    """

    def __init__(self, dataset_val, images_pool):
        self.dataset_val = dataset_val
        self.images_pool = images_pool


    def max_scroe_strategy(self):
        """
        Choosing the images which has lower prediction score(<0.8)
        """
        # for result in resultOfDetection:

        pass

    def min_detection_strategy(self, init_model_infos):
        """
        Choosing the images which has missed detection for instance
        """
        model_folder = 'min_detection_model'
        result = self.detection(init_model_infos)
        # here add some methods to make select more 'clever'
        rank_hard_images = sorted(result.keys())
        total_amount = 30
        # Select most hard images (30 as a step)
        # Start training with select images
        while len(rank_hard_images) >= 30:
            al_model = TrainingProcess()
            al_model_data = rank_hard_images[:30]
            total_amount += 30
            al_model_info = [model_folder, '%images_model' % total_amount]
            al_model.train_model(al_model_data, al_model_info, self.dataset_val)
            al_model.mAP_of_model(al_model_info, self.dataset_val)
            result = self.detection(al_model_info, al_model_data)
            rank_hard_images = sorted(result.keys())
        print("Ending selection")


    def remove_trained_images(self, img_pool, img_list):
        for img in img_list:
            img_pool.remove(img)
        return img_pool


    def max_entropy_strategy(self):
        pass

    def max_gradient_decent(self):
        pass


    def detection(self, model_infos, trained_images=None):
        """
        Try to detect all the rest 270 images by the first version model
        model_info: [model_folder, model_weights_name] first version model which was trained with package1
        imagesPool: [path_to_packges, packges_list] package2 - package10
        """
        # Index of the class in the list is its ID. For example, to get ID of
        class_names = ['BG', 'red_s', 'red_m', 'red_l', 'yellow_s', 'yellow_m', 'yellow_l', 'green_s', 'green_m',
                       'green_l', 'blue_s', 'blue_m', 'blue_l', 'orange_s', 'orange_m', 'orange_l']
        config = ShapesConfig()
        detect_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config, model_info=model_infos)
        # Load weights trained on current model
        cur_model_path = os.path.join(model_infos[0], model_infos[1]+'.h5')
        cur_model = os.path.join(MODEL_DIR, cur_model_path)
        detect_model.load_weights(cur_model, by_name=True)
        # Traverse all the packages(the pool)
        result_of_detection = {}
        for package in self.images_pool:
            image_dir = os.path.join(DATA_DIR, package)
            images_in_package = os.listdir(image_dir)
            for img in images_in_package:
                if trained_images:
                    if img in trained_images:
                        continue
                image = skimage.io.imread(os.path.join(image_dir, img), as_gray=False)
                # Run detection
                results = detect_model.detect([image], verbose=0)
                # Visualize results
                r = results[0]
                # Visualizing the detection result and save it
                # visualize.display_instances(item, image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                # use dict to save the info of the detected instances of each images
                result_of_detection[img] = len(r['scores'])
        print("+++++++detection finished")
        return result_of_detection


if __name__ == "__main__":

    # Validation dataset only need to be loaded once
    val_dataset_root_path = val_img_folder = "val_data"
    val_mask_folder = "mask/val_data"
    val_imglist = os.listdir(val_img_folder)
    val_count = len(val_imglist)
    dataset_val = BlockDataset()
    dataset_val.load_shapes(val_count, val_img_folder, val_mask_folder, val_imglist, val_dataset_root_path)
    dataset_val.prepare()
    print('=====>validation data finished')

    init_model_info = ['30_model_v1', 'best_weights']
    package_list = ['package2', 'package3', 'package4', 'package5', 'package6', 'package7', 'package8', 'package9', 'package10']

    if not init_model_info:
        # Training the first version model
        init_model = TrainingProcess()
        training_images_list = init_model.set_training_data()
        # Set the weights' name and save dir
        # firstModelInfo: [model_folder, model_weights_name] -> str
        init_model_info = ['FirstVersionModel', 'bestModelWeights']
        init_model.train_model(training_images_list, init_model_info, dataset_val)
        init_model.mAP_of_model(first_model_info, dataset_val)

    al_model_result = ActiveLearningStrategy(dataset_val=dataset_val, images_pool=package_list)
    al_model_result.min_detection_strategy(init_model_info)




