# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()

def parse_args1():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--input', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--out', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')


    return parser.parse_args()

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.out)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = glob.glob(os.path.join(args.out, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    # not compute grad infor
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file  wendy change
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            depth = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "deNpy\{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                print(metric_depth)
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "disNpy\{}_disp.npy".format(output_name))
                print(scaled_disp)
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # wendy for depth
            depth_resized_np = depth.squeeze().cpu().numpy()*100
            # wendy skip
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_im = os.path.join(output_directory, "depth\{}.png".format(output_name))
            # # wendy add
            # grey_im = linear_map(depth_resized_np, 0, 20)
            # grey_im_1 = grey_im*255
            # grey_im_11 = (grey_im*255).astype(np.uint8)
            # grey_im_o0 = disp_resized_np * 255
            # grey_im_o = (disp_resized_np*255).astype(np.uint8)
            im = pil.fromarray((depth_resized_np*10).astype(np.uint8))
            im.save(name_dest_im)
            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))

    print('-> Done!')


def generate_optical_flow(args):
    # 创建输出文件夹
    # os.makedirs(output_folder, exist_ok=True)
    # 获取文件列表并按文件名排序
    file_list = sorted(os.listdir(args.input))
    args_s = argparse.Namespace()
    args_s.model_name = args.model
    args_s.no_cuda = args.no_cuda
    args_s.pred_metric_depth = args.pred_metric_depth
    args_s.ext = args.ext
    args_s.out = args.out
    for i in range(len(file_list)):
        args_s.image_path = args.input + '/' + file_list[i]
        test_simple(args_s)


# 线性映射
def linear_map_matrix(matrix, min, max):
    # 找到矩阵中的最小值和最大值
    min_value = min
    max_value = max

    # 确保最大值和最小值不相等，避免除以0的情况
    if max_value == min_value:
        max_value += 1

    # 计算线性映射的值
    mapped_matrix = (matrix - min_value) * 20 / (max_value - min_value)

    return mapped_matrix.astype(np.uint8)


# 最大最小归一化
def min_max_normalize_with_outliers(arr, lower_percentile=0, upper_percentile=100):
    # 计算上下分位数对应的值
    lower_bound = np.percentile(arr, lower_percentile)
    upper_bound = np.percentile(arr, upper_percentile)

    # 剔除超出范围的异常值点，并进行最小-最大归一化
    arr_clipped = np.clip(arr, lower_bound, upper_bound)
    normalized_arr = (arr_clipped - lower_bound) / (upper_bound - lower_bound)
    return normalized_arr

if __name__ == '__main__':
    args = parse_args1()
    # test_simple(args)
    generate_optical_flow(args)
