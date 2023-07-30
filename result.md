1.整个系统的输入只需要rgb图像
[VDO_SLAM](https://github.com/halajun/vdo_slam):


### 生成mask semantic文件夹 mask_rcnn
`tensorflow 1.14.0 和 keras 2.1.6 python 3.75`

2.使用left左目图像生成mask文件夹中的文件 具体运行方法，修改vdomask.py文件中的数据集所在文件夹名称
  使用VDO_SLAM 代码文件夹中的tools\build文件夹中的可执行文件对其mask生成semantic文件夹 第一个参数：输出路径
  eg：./kitti_mask_sem2gt ~/dataset/kitti_056

### 生成depth 文件夹  spsstereo
[spsstereo](https://home.ttic.edu/~dmcallester/SPS/)
3.使用左右双目图像计算视差图，生成在depth文件夹中 具体运行方法 在depth文件夹目录下 运行spsstereo文件夹中的可执行文件 第一个参数是左目的路径 第二个参数是右目图像的路径（修改过代码）
eg：~/code/spsstereo/build/spsstereo ~/dataset/kitti_056/image_0/ ~/dataset/kitti_056/image_1/

### 生成flow 文件夹 PWC-NET
[pwc-net](https://github.com/sniklaus/pytorch-pwc/tree/master)
`cuda + pytorch `

4.使用左目图像生成光流文件在flow中 具体运行方法 在pytorch-pwc文件夹中放置image_0 图像 运行runbatch.py文件

### times文件

需要时间戳文件，在kitti数据集中有，只需要转化格式，运行processtimestamps 可以修改时间戳的格式

----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

**研究ing**

### pose文件

只是用来检测的文件，后边再研究真值怎么或得 

### obj文件

也是用来检验效果的文件，四位对齐的码使用的是语义分割检测出来的结果

### yaml 配置文件是相机的相关参数 在kitti数据集中都有 实际的需要改成相机的实际配置

**下一步**
1.下一步需要看懂代码里是怎么计算速度的 怎么进行 SLAM 工作的 以及SLAM 工作的整体流程 
#2.再找找语义分割 如何在视频里保证它的正确 （主要是这个 现在的语义分割可能还有缺陷 去看看dna那篇）
#3.研究相机的参数
4.逐帧的处理办法
5.一个sh文件可以直接处理所有

