# deeplab
## 数据集
参考：https://www.cnblogs.com/liuwenhua/p/15136361.html
和原来的基本一致
deeplab github: 
1. 需要对中间生成的mask去colormap 使用Deeplab中的model/research/deeplab/dataset中的remove_gt_colormap.py
python remove_gt_colormap.py --original_gt_folder ~/dataset/work_for_mask/train/SegmentationClass -output_dir ~/dataset/work_for_mask/mask

2. 生成索引txt文件
可以暂时将train 和 val 分开在两个目录中只要图片 图片的后缀名不会被记录



3. 在deeplab目录中
python build_voc2012_data.py --image_folder ~/dataset/work_for_mask/data/image/ --semantic_segmentation_folder ~/dataset/work_for_mask/data/mask/ --list_folder ~/dataset/work_for_mask/data/index/ --image_format png --output_dir ~/dataset/work_for_mask/data/tfrecord/
生成tfcord格式

```python
import os,shutil
from PIL import Image

train_path = r'/home/ubuntu/dataset/work_for_mask/train'
filelist_train = sorted(os.listdir(train_path))
val_path = r'/home/ubuntu/dataset/work_for_mask/val'
filelist_val = sorted(os.listdir(val_path))
index_path = r'/home/ubuntu/dataset/work_for_mask/index'

VOC_file_dir = index_path


VOC_train_file = open(os.path.join(VOC_file_dir, "train.txt"), 'w')
VOC_test_file = open(os.path.join(VOC_file_dir, "val.txt"), 'w')
VOC_train_file.close()
VOC_test_file.close()

VOC_train_file = open(os.path.join(VOC_file_dir, "train.txt"), 'a')
VOC_test_file = open(os.path.join(VOC_file_dir, "val.txt"), 'a')

for eachfile in filelist_train:
    (temp_name,temp_extention) = os.path.splitext(eachfile)
    img_name = temp_name
    VOC_train_file.write(img_name + '\n')

for eachfile in filelist_val:
    (temp_name, temp_extention) = os.path.splitext(eachfile)
    img_name = temp_name
    VOC_test_file.write(img_name + '\n')

VOC_train_file.close()
VOC_test_file.close()
```



















# 遇到问题：
1. maskrcnn训练提示：FutureWarning: Input image dtype is bool
修改scikit-image包版本为0.16.2
```
pip install -U scikit-image==0.16.2
```

2. TypeError: Unsupported dtype for TensorType: <dtype: 'int32'>
关掉anaconda 重启
# 数据格式
--cv2_mask:
--+和原图对应名称的png标签名
--json
--+labelme生成的json
--labelme_json
--+和原图对应的文件夹名 eg：000000
--++img.png info.yaml label.png label.names.txt label_viz.png
--pic
--+原图


# 过程
1. labelme 打标签
```
pip install pyqt5==5.15.0
pip install labelme
labelme images --output jsons --nodata --autosave --labels labels.txt
```
其中labels.txt 需要自己写
__ignore__
_background_
uav
2. 使用python批处理数据
```
python pyth.py --input_dir ~/dataset/mask_data1/jsons/ --output_dir ~/dataset/mask_train/ --labels ~/dataset/mask_data1/labels.txt 
python json_to_dataset.py

```


# 查看结果


```python
#!/usr/bin/env python
 
from __future__ import print_function
 
import argparse
import glob
import os
import os.path as osp
import sys
 
import imgviz
import numpy as np
import labelme
 
 
def main(args):
 
    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassnpy"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)
 
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
 
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)
 
        label_file = labelme.LabelFile(filename=filename)
 
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".png")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClassnpy", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".png",
            )
 
        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)[:,:,:3]
 
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)
 
        np.save(out_lbl_file, lbl)
 
        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
#                img=imgviz.rgb2gray(img),
                img,
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../edge_fence_20210203/jsons", type=str, help="input annotated directory")
    parser.add_argument("--output_dir", default="../edge_fence_20210203_voc", type=str, help="output dataset directory")
    parser.add_argument("--labels", default="../edge_fence_20210203/labels.txt", type=str, help="labels file")
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()
    return args
 
 
if __name__ == "__main__":
    args = get_args()
    main(args)
```

```python
import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils
import glob
# 最前面加入导包
import yaml

def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )
    parser = argparse.ArgumentParser()
    ###############################################增加的语句##############################
    # parser.add_argument("json_file")
    parser.add_argument("--json_dir", default="/home/wendy/dataset/mask_data/jsons")
    ###############################################end###################################
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    ###############################################增加的语句##############################
    assert args.json_dir is not None and len(args.json_dir) > 0
    # json_file = args.json_file
    json_dir = args.json_dir
    if osp.isfile(json_dir):
        json_list = [json_dir] if json_dir.endswith('.json') else []
    else:
        json_list = glob.glob(os.path.join(json_dir, '*.json'))
    ###############################################end###################################
    for json_file in json_list:
        json_name = osp.basename(json_file).split('.')[0]
        out_dir = args.out if (args.out is not None) else osp.join(osp.dirname(json_file), json_name)
        ###############################################end###################################
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        data = json.load(open(json_file))
        imageData = data.get("imageData")
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )
        PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
        utils.lblsave(osp.join(out_dir, "label.png"), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))
        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")
        logger.info("Saved to: {}".format(out_dir))
        #######
        #增加了yaml生成部分
        logger.warning('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)
        logger.info('Saved to: {}'.format(out_dir))

if __name__ == "__main__":
    main()

```
训练代码：
```python
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
    IMAGE_MAX_DIM = 2048
    RPN_ANCHOR_SCALES = (16 * 6, 32 * 6, 64 * 6, 128 * 6, 256 * 6)
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50

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

dataset_root_path = "/home/hp/code_gwp/Mask_RCNN/samples/uav/"
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

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=10, layers="all")

```

