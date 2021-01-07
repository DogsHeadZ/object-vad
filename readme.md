# Training command
python train_withoutmem_addeva.py --config configs/vad_baseline.yaml --gpu 0,1,2,3

- 添加yolov5做目标检测，参考https://github.com/yuguangnudt/VEC_VAD结合运动信息提取RoI，代码在`getRoI.py`
- 修改`vad_datasets.py`