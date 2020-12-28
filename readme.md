复现了基于预测进行视频异常检测的2018年论文**Future Frame Prediction for Anomaly Detection – A New Baseline**，FramePred。参考代码https://github.com/fjchange/pytorch_ano_pre


# Training command
`python train_pre_flownet2.py --config configs/vad_pre_flownet2.yaml --gpu 0`

使用flownet2作为光流网络

or

`python train_pre_liteflownet.py --config configs/vad_pre_liteflownet.yaml --gpu 0`

使用liteflownet作为光流网络

目前在单GPU上进行实验，在avenue数据集上能够达到88+的精度，而且这是在光流损失为0的情况下，加上光流损失或许能够逼近原论文作者后来在论文**Margin Learning Embedded Prediction for Video Anomaly Detection with A Few Anomalies**中使用**不同结构**复现的精度89.2。在ped2数据集上的精度不理想，大概为89多一点，原因应该在于原参考代码是在avenue上调的参数，在ped2上不适用。

目前这还是使用了参考代码中的unet结构作为生成器，如果换成原论文作者后来采用的结构精度或许会提高：

![image-20201228094649026](/Users/feihu/Library/Application Support/typora-user-images/image-20201228094649026.png)

其中编码器和解码器采用Cycle-GAN的结构consisting of 3 convolution layers and 6 residual blocks.

