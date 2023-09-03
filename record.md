1. monodepth2 用于对单目深度进行预测
github：https://github.com/nianticlabs/monodepth2
运行方法：
```
cd ~/code/VDO/monodepth2
conda activate G_torch
python test_simple.py --input ./assets/image_0/ --model mono+stereo_640x192 --out ./assets/out/

```
2. pwc_net 用于处理光流信息
```bash
git clone https://github.com/sniklaus/pytorch-pwc.git
依赖：pytorch 
pip install cupy-cuda11x
# 运行
conda activate G_torch
python runbatch.py ~/dataset/act_data/image_0/ ~/dataset/act_data/flow/
```

3. 光流结果查看工具的使用
```
cd ~/code/VDO/flowShow/flow-code
python flow2image.py  ~/dataset//mono_uav/uav_test/flow ~/dataset/mono_uav/uav_test/flow_im
```

4. mask rcnn 处理语义
git clone https://github.com/matterport/Mask_RCNN.git
pip install tensorflow==1.14.0 keras==2.1.6 h5py==2.10.0 protobuf==3.20.0
pip3 install -r requirements.txt
python3 setup.py install

requirements.txt 改成以下
```
numpy
scipy
Pillow
cython
matplotlib
scikit-image
imgaug
IPython[all]
```
运行方式
复制 权重文件到py文件目录下 参数是根目录
python vdomask.py ~/dataset/mono_uav/uav_test/

5. rename 工具使用
```
# 路径 扩展名 要减去的数字
python rename.py ~/dataset/act_data/image_0/ .png 25
```

6. spsstereo 处理双目深度
路径：https://home.ttic.edu/~dmcallester/SPS/
```
# 在深度图所在文件夹内运行
~/code/VDO/spsstereo/build/spsstereo  ~/dataset/airsim/images_0/ ~/dataset/airsim/images_1/

```
7. airsim 数据录制后处理
``` 
# 将数据分开成左目和右目和时间戳文件
python alignTimestampEuRoC.py ~/xxxxxx/airsim_rec.txt
# 处理名称 左右名一样的从小到大排序 7 位 打开python文件改一下路径
python renamebagpng.py 
# 复制想要的数据到 制定目录
cp image_1/00000{20..76}.png image1
# 从0开始重命名 后缀名 初始值
python rename.py ~/dataset/airsim/image_0/ .png 25
# 生成时间戳文件
python proessTime.py ~/share/data/airsim2.0/ 18 99

```

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
数据集
~/dataset/act_data
运行方法：
```
# 先修改 vdo_slam 中Tracking cpp文件中200行的单目或双目模式

cd ~/code/VDO/VDO-SLAM/out
../example/vdo_slam ../example/kitti-0000-0013.yaml ~/dataset/act_data/


```
 
