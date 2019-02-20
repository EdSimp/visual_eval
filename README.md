## evaluation_visulization  
分为三个主要的文件

1. DataLoader为IO的类，实现数据从txt的读入及分段截取数据。输入为txt文件，内部数据的格式为numpy及DataFrame，为定长数据
2. Evaluation为评测的类，实现数据metrics的评测。输入为DataFrame，输出为mse、mse_only等指标，并可以进行top_k轨迹的挑选
3. Visualizaiton为可视化类，实现folder下结果的可视化。输入为一个视频、一个gt的txt、一个预测的txt及infos.json（记录车的数据）



#### 使用方法：

##### 一、Visualization

关联脚本为：Visualizaiton.py、DataLoader.py、util.py、config.py。config脚本中为参数设置，代码运行命令可为：

```python
python Visualization.py --origin_input_base="origin输入地址"  ...
```

可选参数为：

--origin_input_base : A str

--result_input_base : A str

--rd_root : A str

--video_root : A str

--image_root : A str

--save_path : A str

--start_fid : A int

--end_fid : A int

--is_saveimage : A boolean

--save_image_num : A int

--actor :  a str



##### 二、Attribute

关联脚本为：Attributes.py、DataLoader.py、util.py、Evaluation.py

```python
python Attributes.py
```

