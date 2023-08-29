# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
iter_num = 0
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1536
    RPN_ANCHOR_SCALES = (48, 64, 72, 96, 128)
    #RPN_ANCHOR_SCALES = (16 * 6, 32 * 6, 64 * 6, 128 * 6, 256 * 6)
    RPN_ANCHOR_RATIOS = [0.4, 0.5, 0.8]
    TRAIN_ROIS_PER_IMAGE = 6
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.01

config = ShapesConfig()
config.display()

class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels
    
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask
    
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        self.add_class("shapes", 1, "uav")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "/info.yaml"
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "/img.png")
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        count = 1
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("uav") != -1:
                labels_form.append("uav")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

dataset_root_path = "/home/ubuntu/wendy/dataset/uav1/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

dataset_train = DrugDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

dataset_val = DrugDataset()
dataset_val.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
init_with = "coco"

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last()[1], by_name=True)

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=20, layers="all")
