from enum import Flag
import os
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.visualize import display_instances
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


import argparse
import matplotlib.pyplot as plt

class myMaskRCNNConfig(Config):
    NAME = "MaskRCNN_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    DETECTION_NMS_THRESHOLD = 0.3


config = myMaskRCNNConfig()
print("Loading weights for Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
model.load_weights('/home/ubuntu/code/vdo/study_vdo/mask_rcnn_shapes_0100.h5', by_name=True)
class_names = ['bg','uav']
# 路径
gen_path ='./carmera/'


#wendy
rois = {}  # 存储所有帧上检测到的物体的 rois
object_num = 0
current_obj = []
fram = -1
# wendy change r
def filter_and_keep_best_detections(r):
    """
    对每个物体保留与 rois 相交的框中检测分数最高的框。

    Parameters:
        r (dict): 检测结果字典.
    
    Returns:
        dict: 修改后的检测结果字典.
    """
    new_rois = []
    new_masks = []
    new_class_ids = []
    new_scores = []

    # 对每个检测结果进行处理
    for roi, mask, class_id, score in zip(r['rois'], r['masks'], r['class_ids'], r['scores']):
        best_overlap = 0
        best_roi_index = None

        # 寻找与当前框相交的 rois
        for i, existing_roi in enumerate(new_rois):
            overlap = calculate_iou(roi, existing_roi)
            if overlap > best_overlap:
                best_overlap = overlap
                best_roi_index = i

        if best_roi_index is not None:
            # 如果存在与当前框相交的 rois，检查分数并保留最高分数的框
            if score > new_scores[best_roi_index]:
                new_rois[best_roi_index] = roi
                new_masks[best_roi_index] = mask
                new_class_ids[best_roi_index] = class_id
                new_scores[best_roi_index] = score
        else:
            # 如果不存在与当前框相交的 rois，添加当前框到新的列表中
            new_rois.append(roi)
            new_masks.append(mask)
            new_class_ids.append(class_id)
            new_scores.append(score)

    # 更新检测结果
    r['rois'] = np.array(new_rois)
    r['masks'] = np.array(new_masks)
    r['class_ids'] = np.array(new_class_ids)
    r['scores'] = np.array(new_scores)
    return r


#wendy caclulate iou
def calculate_iou(roi1, roi2):
    # 计算两个 ROIs 的交集
    x1 = max(roi1[0], roi2[0])
    y1 = max(roi1[1], roi2[1])
    x2 = min(roi1[2], roi2[2])
    y2 = min(roi1[3], roi2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # 计算两个 ROIs 的并集
    area1 = (roi1[2] - roi1[0] + 1) * (roi1[3] - roi1[1] + 1)
    area2 = (roi2[2] - roi2[0] + 1) * (roi2[3] - roi2[1] + 1)
    
    union = area1 + area2 - intersection
    
    # 计算交/小
    iou = intersection / min(area1,area2)
    # 交比并 
    # iou = intersection /union
    
    return iou

#wendy caclulate x,y change displacement
def calculate_displace(roi1,roi2):
    # h1  = roi1[3] - roi1[1]
    # h2  = roi2[3] - roi2[1]
    # w1 = roi1[2] - roi1[0]
    # w2 = roi2[2] - roi2[0]
    xl = abs(roi1[0]-roi2[0])/100
    xr = abs(roi1[2]-roi2[2])/100
    yt = abs(roi1[1]-roi2[1])/100
    yb = abs(roi1[3]-roi2[3])/100
    # sim= abs(h2-h1)/(h1+h2)+abs(w2-w1)/(w1+w2)
    # if(sim<0.1):
    #     a = 0.6
    #     b = 0.4
    # else:
    #     a = 0.4
    #     b = 0.6
    res =0.65*calculate_iou(roi1,roi2)+0.35*(1-(xl+xr+yt+yb))
    #print(roi1,roi2,res)
    return res
    



    # 算法思想： 如果两个框的大小差不多 就计算他们的位移差 ，四个位移差值相近即同一物体 这里应该选取一个位移（一个大的百分比）
    # 如果大小相差很大，计算边的比值，如果相差很小有可能是同一物体，如果相差很大就不是同一物体（返回百分比）
    # 第一步计算框的大小



#wendy  find best matching return best match  [curr_index,rois_dict.key]  then input (dic,maskrcnn r)
def match_rois(dict_A, mask_r,threshold):
    match_indices = []
    vist_k = {key: False for key in range(1, len(dict_A)+1)}
    for roi,i in zip(mask_r['rois'],range(len(r['class_ids']))):
        best_overlap = threshold
        best_match_index = -1
        for k in dict_A:
            overlap = calculate_displace(roi,dict_A[k][0])
            # print("key:",k,"i",i,"overlap",overlap,"dic",dict_A[k][1],"curr",class_names[r['class_ids'][i]],"curr",roi,"dict",dict_A[k][0])
            if not vist_k[k] and overlap > best_overlap and class_names[r['class_ids'][i]] == dict_A[k][1]:
                best_overlap = overlap
                best_match_index = k
        vist_k[best_match_index] = True
        match_indices.append([i,best_match_index])
    return match_indices

#wendy there need a function to update rois dictionary 
def update(match_indices,mask_r):
    current_obj = []
    for n, o in match_indices:
        if o != -1:
            #print(o,mask_r['rois'][n])
           # print(rois)
            rois[o][0] = mask_r['rois'][n]
            rois[o][1] = class_names[mask_r['class_ids'][n]]
            current_obj.append(o)
        else:
            rois[len(rois)+1] = [mask_r['rois'][n],class_names[mask_r['class_ids'][n]]]
            current_obj.append(len(rois))
    return current_obj
    
# wendy convert mask
def convert_mask(mask):
    height, width, num_objects = mask.shape
    converted_mask = np.zeros((height, width), dtype=np.int32) - 1

    for i in range(num_objects):
        object_mask = mask[:, :, i]
        converted_mask[object_mask == 1] =current_obj[i]

    return converted_mask

def tar_all(gen_path):
    input_folder = gen_path + 'image_0'  # 输入文件夹路径
    output_folder =gen_path + 'mask'  # 输出文件夹路径
    obj_name = gen_path + 'object_pose.txt'
    # 在 tar_all 函数开头添加这段代码
    mask_img_folder = os.path.join(gen_path, 'mask_img')
    os.makedirs(mask_img_folder, exist_ok=True)

    global fram,current_obj,object_num,r
    for filename in sorted(os.listdir(input_folder)):
        fram += 1
        if filename.endswith('.png'):
            # 构建输入图片的完整路径
            image_path = os.path.join(input_folder, filename)
            #print(filename)

            # 读取输入图片
            image = load_img(image_path)
            image = img_to_array(image)
            # 进行预测
            results = model.detect([image], verbose=0)
            r = results[0]
           # r = filter_and_keep_best_detections(r)
            classes = r['class_ids']
            #print("Total Objects found:", len(classes))
            """ for i in range(len(classes)):
                print(class_names[classes[i]]) """
            
            # 可视化语义分割结果
            # display_instances(image,r['rois'],r['masks'],r['class_ids'],class_names,r['scores'])
            # 可视化语义分割结果
            fig, ax = plt.subplots()
            display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=ax)
            
            # 保存可视化结果图像
            mask_show_filename = filename + '.visual.png'
            mask_show_path = os.path.join(mask_img_folder, mask_show_filename)
            
            canvas = FigureCanvas(fig)
            canvas.print_png(mask_show_path)
            plt.close(fig)  # 关闭图像以防止显示

 
            # 将检测到的物体的 rois 存储到数组中
            # 检测和上帧中储存障碍物的距离 重构 字典
            if not rois: 
                new_rois = {}
                for roi,_i in zip(r['rois'],range(len(classes))):
                    object_num += 1
                    current_obj.append(object_num)
                    new_rois[object_num] = [roi,class_names[classes[_i]]]
                rois.update(new_rois)
                with open(obj_name,"w") as file:
                    for curr_o in current_obj:
                        file.write(f"{fram}\t{curr_o}\t{rois[curr_o][0][1]}\t{rois[curr_o][0][0]}\t{rois[curr_o][0][3]}\t{rois[curr_o][0][2]}\t{0}\t{0}\t{0}\t{0}\n")
            else:
                match_res = np.array(match_rois(rois,r,0.3))
                current_obj = update(match_res,r)
                with open(obj_name,"a") as file:
                    for curr_o in current_obj:
                        file.write(f"{fram}\t{curr_o}\t{rois[curr_o][0][1]}\t{rois[curr_o][0][0]}\t{rois[curr_o][0][3]}\t{rois[curr_o][0][2]}\t{0}\t{0}\t{0}\t{0}\n")
                
                
            print(filename,current_obj)

            # 构建输出文件的完整路径
            output_filename = filename + '.mask'
            output_path = os.path.join(output_folder, output_filename)

            # 保存结果
            with open(output_path, 'w') as file:
                height, width,_nu = r['masks'].shape
                num_objects =len(rois)

                file.write(f'{height} {width} {num_objects}\n')
                for i in range(1,len(rois)+1):
                    file.write(rois[i][1] + '\n')
                np.savetxt(file, convert_mask(r['masks']), fmt='%d', delimiter=' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mask")
    parser.add_argument("gen", type=str, help="根目录")
    args = parser.parse_args()
    tar_all(args.gen)
