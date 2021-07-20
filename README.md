# Yangon Chickenfish

本项目是一个基于YOLOv4的卫星图像目标体识别与集群检测模型。目前应用于检测分析缅甸仰光周边鸡舍+鱼塘养殖模式的发展情况，同时也可以迁移至其他目标检测场景。

- 参考的PyTorch+YOLOv4实现：https://github.com/WongKinYiu/PyTorch_YOLOv4
- 下载训练数据（放入detection）：https://pan.baidu.com/s/1Pa_Yv9ozAoSmdu5IhJ_u1w 提取码：ztmj
- 下载预训练模型：https://pan.baidu.com/s/1hz3vQF-WoDfvsFNb7k7BHQ 提取码：93ur



## 1 安装

```
pip install -r requirements.txt
```

- 若需要使用Mish激活函数，需先安装mish-cuda：https://github.com/thomasbrandon/mish-cuda



## 2 目标检测

目标检测模型位于detection目录下。此后所有操作与指令均在detection目录下进行。

### 2.1 数据准备

- 将图片数据集整理成detection/images，将VOC标准的所有标注文件放入detection/annotations，保证标注文件名除拓展名外与对应图片一致，且每个标注均存在对应的图片；
- 修改参数并依次运行1.split_data.py 2.get_label.py；
- 如果需要进行预测，修改参数并运行3.get_detect_list.py。

### 2.2 训练

根据需要修改train.sh以及cfg目录下的参数，然后运行train.sh。训练后会在run目录下创建一个子目录，含有训练后的网络权重和训练过程数据记录。

### 2.3 测试

根据需要修改test.sh以及cfg目录下的参数，然后运行test.sh。测试指标包括在不同的iou_thresh（即区分正确预测和错误预测时，采用的预测框与标注框的IOU的阈值）下的P(precision)，R(recall)，mAP等。

### 2.4 预测

根据需要修改detect.sh以及cfg目录下的参数，然后运行detect.sh。预测将在inference目录下输出预测结果。



## 3 集群检测

集群检测模型位于cluster目录下。此后所有操作与指令均在cluster目录下进行。

### 3.1 数据准备

数据需要放至data目录下，格式为txt，内容格式如下：

```
x0 y0
x1 x1
x2 y2
```

即每行代表一个目标点的坐标。可以先在raw目录下进行预处理。可以参考raw/process_10-18.py。

### 3.2 集群检测

根据需要修改并运行detect.py。会在result目录下生成txt格式的集群检测结果。格式如下：

```
cluster 
3 1 1 2 2
1 1
2 2
1.5 1.5
cluster
4 3 4 5 6
3 6
5 4
4.5 5
4 5
```

解释：cluster表示此后内容为一个集群；下一行五个数分别为size xmin ymin xmax ymax，标记集群的大小和边界；然后其下size行是构成集群的所有目标的坐标。

集群检测的规则：设定距离阈值x; 对每个物体DFS搜索周围x距离内的物体，如存在则归为同一集群。最后集群大小大于k的会被保留.

### 3.3 可视化

可视化指生成前一过程检出的集群区域的对应卫星图像。需要首先有整片区域卫星图像数据，以及detect.py的运行结果。

visualize.py中含有的可视化功能包括：

- 输出任意经纬度范围，任意清晰度的地图（getmap）
- 输出检测目标的截图，支持随机采样/全部输出（mapcut_single)
- 将检测目标在模型的输入图片上进行标注，支持随机采样/全部输出（annotate_single）
- 输出集群的截图（mapcut_cluster)
- 对集群按地理位置进行编号并对每个集群作其在多年间的变化图（comparison）
- 生成集群和目标的热度图（heatmap_single单目标，heatmap_cluster集群，heatmap_region任意区域）

其中前五个在result目录下生成结果。



 

