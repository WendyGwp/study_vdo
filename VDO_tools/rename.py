# -*- coding: utf-8 -*-
import os
import argparse

def rename(folder_path, backname, num):
    # 文件夹是否存在
    if os.path.isdir(folder_path):
        # 获取指定扩展名的文件
        files = [file for file in os.listdir(folder_path) if file.lower().endswith(backname)]
    
        for file in files:
            # 提取文件扩展名之前的最后三个字符
            last_three_chars = file[-7:-4]
            
            try:
                last_three_digits = int(last_three_chars)
                new_digits = last_three_digits - num
                name_s = "{:06d}" + backname
                new_name = name_s.format(new_digits)
                old_path = os.path.join(folder_path, file)
                new_path = os.path.join(folder_path, new_name)
                
                os.rename(old_path, new_path)
                
                print(f"已将文件 '{file}' 重命名为 '{new_name}'")
            except ValueError:
                print(f"文件名: {file}，后三位不是数字")
    else:
        print(f"文件夹 '{folder_path}' 不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重命名图片文件")
    parser.add_argument("folder_path", type=str, help="图片文件所在的文件夹路径")
    parser.add_argument("backname", type=str, help="图片文件的扩展名，例如：'.png'")
    parser.add_argument("num", type=int, help="重命名时使用的数字")

    args = parser.parse_args()

    rename(args.folder_path, args.backname, args.num)

