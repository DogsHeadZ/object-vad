# Training command
python train_withoutmem_addeva.py --config configs/vad_baseline.yaml --gpu 0,1,2,3

- 添加yolov5做目标检测，参考https://github.com/yuguangnudt/VEC_VAD结合运动信息提取RoI，代码在`getRoI.py`
- 修改`vad_datasets.py`



# 2021.1.9

增加dataloader，重命名了一些文件

- dataloader：5帧图像

- dataloader_object: 一个sample返回一个目标

- dataloader_frameobject: 一个sample返回图像上所有目标的图像和光流

  ```
  return batch, batch_flow   
  #最后即返回这段视频片段为batch， 大小为[目标个数, _time_step+num_pred, 图片的通道数, _resize_height, _resize_width]
  光流为 batch_flow 大小为[目标个数, _time_step+num_pred-1, 图片的通道数, _resize_height, _resize_width]
  ```

增加光流计算：getFlow.py



### 一些非常amazing的东西

原来的yolo v5在cuda：0上可以正常运行，但是其他卡或者多卡的时候有点问题，重新在https://github.com/ultralytics/yolov5下了最新版本

Flownet2在cuda：0和其他卡上都能跑，但是cuda：0的光流是正常的，其他都不太对劲，而且如果不是在cuda：0上跑，大概计算了103张图的光流之后会报错，这个问题很魔幻，还没解决



# 2021.1.10

完善了dataloader中光流的读取

getFlow.py保存光流的时候改为直接用torch.save()保存tensor，避免转换成numpy太麻烦，而且维度变换有些问题。



# 2021.1.25

合并了师兄objectloss跑对比实验

### dataloader：

- dataloader：5帧图像
- dataloader_object: 一个sample返回一个目标
- dataloader_frameobject: 一个sample返回图像上所有目标的图像和光流
- dataloader_frameflow: 返回一个sample的图像，bboxes和最后两帧之间的光流（**加入对预提取了光流的处理，感觉训练的时候计算太慢了，而且没解决之前flownet不用cuda:0就有问题的错误**）

**losses.py增加objectloss**

**更新`evaluate.py`，测试的时候用**，示例如下，（如果不需要输出可视化结果，则is_visual可以设成False，并且后面的参数都不用填

```python
frame_AUC, roi_AUC = evaluate(test_dataloader, model, labels_list, videos, loss_func_mse, config['test_dataset_type'], test_bboxes=config['test_bboxes'],
                frame_height = train_dataset_args['h'], frame_width=train_dataset_args['w'], 
                is_visual=True, mask_labels_path = config['mask_labels_path'], save_path = os.path.join(save_path, "./final"), labels_dict=labels) 
```

没有对每段分开的视频做归一化，而是所有的视频一起归一化，因为看了一下数据，有些整段视频都是异常事件。

### 数据标签

之前发现ffp论文提供的代码中标签有点问题，所以根据原始数据集的像素级标注重新生成了标签，标签的格式也有变化。为一个dict，key为`01,02,03`这些，值为对应的视频段中每一帧的标签。



