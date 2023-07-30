import os
import argparse

def convert_flo_to_png(flo_path, png_path):
    flo_files = [file for file in os.listdir(flo_path) if file.lower().endswith('.flo')]
    for flo_file in flo_files:
        png_file = os.path.splitext(flo_file)[0] + '.png'
        flo_file_path = os.path.join(flo_path, flo_file)
        png_file_path = os.path.join(png_path, png_file)
        ml = './color_flow\t' + flo_file_path + '\t' + png_file_path
        os.system(ml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 .flo 文件转换为 .png 文件")
    parser.add_argument("flo_path", type=str, help=".flo 文件所在的文件夹路径")
    parser.add_argument("png_path", type=str, help=".png 文件输出的文件夹路径")

    args = parser.parse_args()

    convert_flo_to_png(args.flo_path, args.png_path)

