# Training command
python train_withoutmem_addeva.py --config configs/vad_baseline.yaml --gpu 0,1,2,3

- 添加yolov5做目标检测，参考https://github.com/yuguangnudt/VEC_VAD结合运动信息提取RoI，代码在`getRoI.py`
- 修改`vad_datasets.py`



2021.1.9

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



## 一些非常amazing的东西

原来的yolo v5在cuda：0上可以正常运行，但是其他卡或者多卡的时候有点问题，重新在https://github.com/ultralytics/yolov5下了最新版本

Flownet2在cuda：0和其他卡上都能跑，但是cuda：0的光流是正常的，其他都不太对劲，而且如果不是在cuda：0上跑，大概计算了103张图的光流之后会报错，这个问题很魔幻，还没解决