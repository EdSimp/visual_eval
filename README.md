## evaluation_visulization  
分为三个主要的文件

1. DataLoader为IO的类，实现数据从txt的读入及分段截取数据。输入为txt文件，内部数据的格式为numpy及DataFrame，为定长数据
2. Evaluation为评测的类，实现数据metrics的评测。输入为DataFrame，输出为mse、mse_only等指标，并可以进行top_k轨迹的挑选
3. Visualizaiton为可视化类，实现folder下结果的可视化。输入为视频、gt的txt、预测的txt及infos.json（记录车的数据）



#### 使用方法：

##### 一、Visualization（可视化）

**功能：**

对数据可视化，输入测试的输入输出folder及对应的txt文件（confluence上定义格式），输出为可视化结果



**关联脚本：**

visualizaiton.py、dataLoader.py、util.py、config.py。config脚本中为参数设置，代码运行命令可为：

```python
python visualization.py --origin_input_base="origin输入地址"  ...
```



**可选参数：**（config里有注释）

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



##### 二、Attribute（属性分析）

**功能：**

对数据属性分析，可使用<u>**属性**</u>为：velocity_mean（平均速度）、velocity_std（速度波动）、distance_mean（平均距离）、distance_std（距离波动）、direction（移动方向：static/left/right/straight）、shake（smooth/jitter）、crowd（5米内的人数）

可使用**<u>画图</u>**为：饼状图（pie）、mse折线图（broken_line）

若不指定除路径外参数，默认将得到所有属性的饼状图及mse折线图



**关联脚本**：

attributes.py、dataLoader.py、util.py、evaluation.py

```python
python attributes.py
```



**可选参数：**

--origin_path：A str，输入txt的文件夹路径

--result_path：A str，输出txt的文件夹路径

--eval_path：A str，保存evaluation的结果的txt路径

--paint_save_path ：A str，画图结果保存的文件路径

--actor：A str，行人（ped）/车辆（veh）

--is_eval_all：A bool，默认为True，即评估所有属性，若需要单独评估属性，改值设为False，并将需要估计的属性写在attr_name中，attr_name为list

--is_group_attr：A bool，默认为False。如果需要自己组合属性，将改值置为True，改代码中设置的组合属性为对（smooth&velocity<3）及~（smooth&velocity<3）的属性进行分析，若有其他需要，可按照例程更改



#### 三、Evaluation（评估）

**功能：**

对数据的到mse及mse_only结果



**关联脚本**：

evaluation.py、dataloader.py、config.py

```
python evaluation.py
```



**可选参数：**（config里有注释）

--origin_input_base： A str，输入txt路径

--result_input_base：A str，输出txt路径

--actor：A str，行人（ped）/车辆（veh）