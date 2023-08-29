import os
import torch
import torchvision.transforms as transforms
from run import estimate
import numpy
import PIL
import PIL.Image
import argparse

def generate_optical_flow(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件列表并按文件名排序
    file_list = sorted(os.listdir(input_folder))

    # 定义图像变换
    transform = transforms.ToTensor()

    for i in range(len(file_list) - 1):
        image_one = PIL.Image.open(os.path.join(input_folder, file_list[i])).convert("RGB")
        image_two = PIL.Image.open(os.path.join(input_folder, file_list[i+1])).convert("RGB")

        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(image_one)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(image_two)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))


        tenOutput = estimate(tenOne, tenTwo)

        objOutput = open(os.path.join(output_folder, file_list[i+1][:-4]+'.flo'), 'wb')

        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
        numpy.array([tenOutput.shape[2], tenOutput.shape[1]], numpy.int32).tofile(objOutput)
        numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

        objOutput.close()

        print(f"Generated optical flow for {file_list[i+1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optical flow from image sequence")
    parser.add_argument("input_folder", type=str, help="Folder containing input images")
    parser.add_argument("output_folder", type=str, help="Folder to save generated optical flow")

    args = parser.parse_args()

    generate_optical_flow(args.input_folder, args.output_folder)
