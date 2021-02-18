  
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data

from getROI import *
from getFlow import *

import argparse

import sys
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
import sys
sys.path.insert(0, './yolov5')

from flownet2.models import FlowNet2  # the path is depended on where you create this module
from flownet2.utils_flownet2.frame_utils import read_gen  # the path is depended on where you create this module
from flownet2.utils_flownet2 import flow_utils, tools

import torchvision.transforms as transforms
import torch.nn.functional as F


def np_load_frame_roi(filename, resize_height, resize_width, bbox):
    
    (xmin, ymin, xmax, ymax) = bbox 
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def roi_flow(frame_flow, bbox, resize_height, resize_width, img_size):
    # 坐标转换
    frame_flow = frame_flow.cpu()
    (xmin, ymin, xmax, ymax) = bbox
    heigh_coef = frame_flow.shape[1]/img_size[0]
    width_coef = frame_flow.shape[2]/img_size[1]
    # print("img_size: ", img_size)
    # print(frame_flow.shape)

    xmin = int(xmin*width_coef)
    ymin = int(ymin*heigh_coef)
    xmax = int(xmax*width_coef)
    ymax = int(ymax*heigh_coef)
    # print(xmin, ymin,xmax, ymax)

    # roi_flow = frame_flow[ymin:ymax, xmin:xmax, :]
    # print(roi_flow.shape)
    roi_flow = frame_flow[: , ymin:ymax, xmin:xmax] 
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # roi_flow = transform(roi_flow)

    # print(roi_flow.shape)
    # 没有办法直接resize 所以使用双线性插值
    roi_flow = roi_flow.unsqueeze(0)
    roi_flow = F.interpolate(roi_flow, size=([resize_width,resize_height]), mode='bilinear', align_corners=True)
    roi_flow = roi_flow.squeeze(0)

    return roi_flow


class VadDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, dataset='', time_step=4, num_pred=1,  bbox_folder=None, device=None, flow_folder=None):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred

        self.dataset = dataset #ped2 or avenue or ShanghaiTech

        self.bbox_folder = bbox_folder  #如果box已经预处理了，则直接将npy数据读出来, 如果没有，则在get_item的时候计算
        if bbox_folder == None: #装载yolo模型
            self.yolo_weights = 'yolov5/weights/yolov5s.pt'
            self.yolo_device = device
            self.yolo_model = attempt_load(self.yolo_weights, map_location=self.yolo_device)  # load FP32 model

        self.flow_folder = flow_folder
        if self. flow_folder == None: #装载flownet
            parser = argparse.ArgumentParser()
            parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
            parser.add_argument("--rgb_max", type=float, default=255.)            
            args = parser.parse_args()
            
            self.device = device
            self.flownet = FlowNet2(args).to(self.device)
            dict_ = torch.load("flownet2/FlowNet2_checkpoint.pth.tar")
            self.flownet.load_state_dict(dict_["state_dict"])


        self.setup()
        self.samples = self.get_all_samples()
        


    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            video_name = os.path.split(video)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))   #每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])    #每个目录下视频帧的个数
        
        video_name = os.path.split(videos[0])[-1]
        self.img_size = cv2.imread(self.videos[video_name]['frame'][0]).shape # [w, h, c]

        
        if self.bbox_folder != None: #如果box已经预处理了，则直接将npy数据读出来
            for bbox_file in sorted(os.listdir(self.bbox_folder)):
                video_name = bbox_file.split('.')[0]
                self.videos[video_name]['bbox'] = np.load(os.path.join(self.bbox_folder, bbox_file), allow_pickle=True) # 每个目录下所有视频帧预提取的bbox

        if self.flow_folder != None: #如果已经预处理了，直接读取
            for flow_dir in sorted(os.listdir(self.flow_folder)):
                video_name = flow_dir
                path = os.path.join(self.flow_folder, flow_dir)
                self.videos[video_name]['flow'] = sorted(glob.glob(os.path.join(path, '*')))
        # print(self.videos[video_name]['flow'])

                    


    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):     #减掉_time_step为了刚好能够滑窗到视频尾部
                frames.append(self.videos[video_name]['frame'][i])          #frames存储着训练时每段视频片段的首帧，根据首帧向后进行滑窗即可得到这段视频片段
                           
        return frames               
            
        
    def __getitem__(self, index):
        # video_name = self.samples[index].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        # frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        # windows
        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int( os.path.split(self.samples[index])[1].split('.')[-2] )

        if self.bbox_folder != None : #已经提取好了bbox，直接读出来
            bboxes = self.videos[video_name]['bbox'][frame_name]
        else: #需要重新计算
            frame = self.videos[video_name]['frame'][frame_name]
            frames = [self.videos[video_name]['frame'][frame_name+i] for i in range(self._time_step+self._num_pred)]
            bboxes = RoI(frames, self.dataset, self.yolo_model, self.yolo_device)

        if self.flow_folder != None: #已经提取好了，直接读出来
            frame_flows = []
            for i in range(self._time_step+self._num_pred-1):
                # frame_flow = np.load(self.videos[video_name]['flow'][frame_name+i], allow_pickle=True)
                frame_flow = torch.load( self.videos[video_name]['flow'][frame_name+i] )
                frame_flows.append( frame_flow )
        else:
            frame_flows = []
            for i in range(self._time_step+self._num_pred-1):
                frame_flow = get_frame_flow(self.videos[video_name]['frame'][frame_name+i], self.videos[video_name]['frame'][frame_name+i+1], self.flownet, self.device, 512, 384)
                frame_flows.append(frame_flow)
        
        batch = []
        batch_flow = [] 
        for bbox in bboxes:
            # img
            object_batch = []
            for i in range(self._time_step+self._num_pred):
                image = np_load_frame_roi(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width, bbox)   #根据首帧图片名便可加载一段视频片段
                if self.transform is not None:
                    object_batch.append(self.transform(image))
            object_batch = torch.stack(object_batch, dim=0)
            # print("object_batch.shape: ", object_batch.shape)
            batch.append(object_batch)

            
            # flow
            object_flow = []
            for i in range(self._time_step+self._num_pred - 1):
                flow = roi_flow(frame_flows[i], bbox, self._resize_height, self._resize_width, self.img_size)
                object_flow.append(flow)
            object_flow = torch.stack(object_flow, dim=0)
            batch_flow.append(object_flow)



        batch = torch.stack(batch, dim=0)
        batch_flow = torch.stack(batch_flow, dim=0)
        
        # print(batch.shape, batch_flow.shape)
        return batch, batch_flow   
        #最后即返回这段视频片段为batch， 大小为[目标个数, _time_step+num_pred, 图片的通道数, _resize_height, _resize_width]
        #                   光流为 batch_flow 大小为[目标个数, _time_step+num_pred-1, 图片的通道数, _resize_height, _resize_width]
                            
        
        
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # test dataloader
    import torchvision
    from torchvision import datasets
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt     
    import torchvision.transforms as transforms

    batch_size = 1
    datadir = "/data0/lyx/VAD_datasets/ped2/testing/frames"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # flow 和 yolo 在线计算
    train_data = VadDataset(video_folder= datadir, bbox_folder = None, dataset="ped2", flow_folder=None,
                            device=device, transform=transforms.Compose([transforms.ToTensor()]), 
                            resize_height=256, resize_width=256)
    
    # # 使用保存的.npy
    # train_data = VadDataset(video_folder= datadir, bbox_folder = "./bboxes/ped2/test", flow_folder="./flow/ped2/test",
    #                         transform=transforms.Compose([transforms.ToTensor()]), 
    #                         resize_height=256, resize_width=256)
    
    # # 仅在线计算flow
    # train_data = VadDataset(video_folder= datadir, bbox_folder = "./bboxes/ped2/test", flow_folder=None, 
    #                         transform=transforms.Compose([transforms.ToTensor()]), 
    #                         resize_height=256, resize_width=256, device = device)
    
    # 仅在线计算yolo
    # train_data = VadDataset(video_folder= datadir, bbox_folder = None,dataset='ped2', 
    #                         flow_folder="./flow/ped2/test",
    #                         transform=transforms.Compose([transforms.ToTensor()]), 
    #                         resize_height=256, resize_width=256, 
    #                         device = device)

    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    unloader = transforms.ToPILImage()

    X, flow = next(iter(train_loader))
    print(X.shape, flow.shape)


    # # 显示一个batch, pil显示的颜色是有问题的，只是大概看一下
    # index = 1
    # for i in range(X.shape[1]):
    #     for j in range(X.shape[2]):
    #         plt.subplot(X.shape[1], X.shape[2], index)
    #         index += 1
    #         print(i,j)
    #         # img = X[j,i*3:i*3+3].cpu().clone()
    #         img = X[0,i,j,:,:,:].cpu().clone()
    #         img = img.squeeze(0) 
    #         img = unloader(img)           
    #         plt.imshow(img)  
    # plt.show()   

