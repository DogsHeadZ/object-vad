import numpy as np
from collections import OrderedDict
import os
import sys
import glob
import cv2
import torch.utils.data as data

from detect_yolo_simple import detect

def np_load_frame_object(filename, loc, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1]. Not normalized here   ---zwh

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    img_crop = image_decoded[loc[1]:loc[3], loc[0]:loc[2]]
    image_resized = cv2.resize(img_crop, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0

    return image_resized


class VadDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))   #每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])    #每个目录下视频帧的个数

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):     #减掉_time_step为了刚好能够滑窗到视频尾部
                frames.append(self.videos[video_name]['frame'][i])          #frames存储着训练时每段视频片段的首帧，根据首帧向后进行滑窗即可得到这段视频片段
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        median_image_path = self.videos[video_name]['frame'][frame_name+1]
        detection = detect(median_image_path)
        objects = []
        for i in range(self._time_step+self._num_pred):
            path = self.videos[video_name]['frame'][frame_name+i]
            channels_per_img = []
            img = cv2.imread(path)
            for j, (*xyxy, conf, cls) in enumerate(reversed(detection)):
                loc = [int(x.item()) for x in xyxy]
                img_crop = img[loc[1]:loc[3], loc[0]:loc[2]]
                image_resized = cv2.resize(img_crop, (self._resize_height, self._resize_width))
                # img_object = np_load_frame_object(path, loc, self._resize_height, self._resize_width)
                if self.transform is not None:
                    channels_per_img.append(self.transform(image_resized))
            channels_per_img = np.concatenate(channels_per_img, axis=0)
            objects.append(channels_per_img)
        return objects     # 最后即返回这段视频片段大小为[（3*单张图片中目标的个数）*图片的通道数, _resize_height, _resize_width]
        
        
    def __len__(self):
        return len(self.samples)
