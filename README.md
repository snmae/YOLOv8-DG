说明： 本项目支持YOLOv8的对应的package的版本是：ultralytics-8.0.0
1.YOLOv8的相关资源
（1）YOLOv8 Github: https://github.com/ultralytics/ultralytics
（2）YOLOv8文档： https://v8docs.ultralytics.com/
2.YOLOv8的环境安装，可以参考我这里的环境安装方式
#pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ultralytics==0.0.59
#pip install -e ultralytics
pip install ultralytics
3.自我创建数据集，在我的论文当中包含了自建数据集的操作，原图后续会在github当中发布，如果想要我标注完的，也可以通过邮箱进行联系。
4.构建自己的1训练集配置文件和模型配置文件（以最基础的yolov8模型架构为例）
#yolov8s.yaml
# Parameters
nc: 4  # number of classes
depth_multiple: 0.33  # scales module repeats
width_multiple: 0.50  # scales convolution channels

# YOLOv8.0s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0s head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 13

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
数据集配置文件
#score_data.yaml

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
train: ....(省略处为自己电脑的文件路径）/images/train # train images
val: ....(省略处为自己电脑的文件路径）/images/val # val images
#test: ....(省略处为自己电脑的文件路径）/images/test # test images (optional)

# Classes（修改为自己数据集标注完的类别）
names:
  0: person
  1: cat
  2: dog
  3: horse
5.yolov8目标检测任务训练，超参数配置
yolo task=detect mode=train model=yolov8s.yaml  data=score_data.yaml epochs=300 batch=16 imgsz=640 pretrained=False optimizer=SGD 
