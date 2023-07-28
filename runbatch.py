import os
import torch
import torchvision.transforms as transforms
from run import estimate
import numpy
import PIL
import PIL.Image
def generate_optical_flow(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件列表并按文件名排序
    file_list = sorted(os.listdir(input_folder))

    # 定义图像变换
    transform = transforms.ToTensor()

    tenOne = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(PIL.Image.open(input_folder + "/" + file_list[0]))[:, :, ::-1].transpose(2, 0, 1).astype(
            numpy.float32) * (
                1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(PIL.Image.open(input_folder + "/" + file_list[0]))[:, :, ::-1].transpose(2, 0, 1).astype(
            numpy.float32) * (
                1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)

    objOutput = open(output_folder + '/' + file_list[0][:-4] + '.flo', 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
    numpy.array([tenOutput.shape[2], tenOutput.shape[1]], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

    objOutput.close()
    for i in range(len(file_list) - 1):
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(
            numpy.array(PIL.Image.open(input_folder+"/"+file_list[i]))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                        1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(
            numpy.array(PIL.Image.open(input_folder+"/"+file_list[i+1]))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                        1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo)

        objOutput = open(output_folder+'/'+file_list[i+1][:-4]+'.flo', 'wb')

        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
        numpy.array([tenOutput.shape[2], tenOutput.shape[1]], numpy.int32).tofile(objOutput)
        numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

        objOutput.close()


        # # 读取当前帧和下一帧图像
        # current_frame = Image.open(os.path.join(input_folder, file_list[i]))
        # next_frame = Image.open(os.path.join(input_folder, file_list[i+1]))
        #
        # # 进行图像变换
        # current_frame = transform(current_frame)
        # next_frame = transform(next_frame)
        #
        # # 使用光流估计函数计算光流
        # flow = estimate(current_frame, next_frame)
        #
        # # 生成输出文件路径
        # flow_output_path = os.path.join(output_folder, f"{file_list[i+1][:-4]}.flo")
        #
        # # 将光流保存为文件
        # torch.save(flow, flow_output_path)

        print(f"Generated optical flow for {file_list[i+1]}")

# 示例用法
input_folder = "./image_0"
output_folder = "./flow"

generate_optical_flow(input_folder, output_folder)