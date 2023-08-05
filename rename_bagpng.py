# -*- coding: utf-8 -*-
import os
import shutil
from datetime import datetime

def get_timestamp(filename):
    try:
        timestamp_str = filename.split(".")[0]
        timestamp = float(timestamp_str)
        return timestamp
    except ValueError:
        return None

def match_and_rename_images(left_folder, right_folder, output_folder_l, output_folder_r):
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    # 获取左右目图像文件名的交集
    common_images = set(left_images) & set(right_images)

    if not os.path.exists(output_folder_l):
        os.makedirs(output_folder_l)
    if not os.path.exists(output_folder_r):
        os.makedirs(output_folder_r)

    # 将匹配的图像文件按照原文件的时间戳排序，并重新命名为00000.png, 00001.png, 00002.png，依次递增
    for index, image_name in enumerate(sorted(common_images, key=get_timestamp)):
        left_image_path = os.path.join(left_folder, image_name)
        right_image_path = os.path.join(right_folder, image_name)

        # 重新命名为00000.png, 00001.png, 00002.png等形式
        new_image_name = "{:07d}.png".format(index)
        output_left_image_path = os.path.join(output_folder_l, new_image_name)
        output_right_image_path = os.path.join(output_folder_r, new_image_name)

        # 复制左右目图像到输出文件夹，并进行重命名
        shutil.copy(left_image_path, output_left_image_path)
        shutil.copy(right_image_path, output_right_image_path)

if __name__ == "__main__":
    left_folder = "/home/ubuntu/dataset/airsim/images_left"  # 左目图像文件夹路径
    right_folder = "/home/ubuntu/dataset/airsim/images_right"  # 右目图像文件夹路径
    output_folder_l = "/home/ubuntu/dataset/airsim/image0"  # 输出文件夹路径
    output_folder_r = "/home/ubuntu/dataset/airsim/image1"
    match_and_rename_images(left_folder, right_folder, output_folder_l, output_folder_r)

