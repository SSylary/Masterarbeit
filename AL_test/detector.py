# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from mrcnn.config import Config
from datetime import datetime

print(tf.__version__)
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "training_data/package10")

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
    NUM_CLASSES = 1 + 15  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    # Use small ROIs Threshold for more detecting
    DETECTION_NMS_CONFIDENCE = 0.5

#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "logs/30_model_v1/best_weights.h5")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Index of the class in the list is its ID. For example, to get ID of
class_names = ['BG', 'red_s','red_m','red_l','yellow_s','yellow_m','yellow_l','green_s','green_m','green_l','blue_s','blue_m','blue_l','orange_s','orange_m','orange_l']
# Load images from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
print('file_names',file_names)
select_images = []
for item in file_names:
    if item == '.directory':
        continue
    print('image',item)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, item),as_gray=False)
    # Run detection
    results = model.detect([image], verbose=0)
    # Visualize results
    r = results[0]
   # print(results)
    print(r['scores'])
    print(r['class_ids'])
   # select the detections with lower scores
    if len(r['scores']) <= 2:
        select_images.append(item)
    else:
        for i in r['scores']:
            # print(type(i))
            if i < 0.8 and item not in select_images:
                select_images.append(item)
print(select_images)

for image_id in select_images:
    im = Image.open(os.path.join(IMAGE_DIR,image_id))
    im.save(os.path.join(ROOT_DIR, 'new_max_score_select_images/{}'.format(image_id)))
    im.close()
