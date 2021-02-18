import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized )/255.0

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
            # video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            video_name = os.path.split(video)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))   #每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])    #每个目录下视频帧的个数

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

        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)   #根据首帧图片名便可加载一段视频片段
            if self.transform is not None:
                batch.append(self.transform(image))
        batch = torch.stack(batch, dim=0)
        # batch = np.concatenate(batch, axis=0)
        # batch = batch.reshape(self._time_step+self._num_pred, -1, batch.shape[-2], batch.shape[-1])
        return batch    #最后即返回这段视频片段大小为[_time_step+num_pred, 图片的通道数, _resize_height, _resize_width]
        
        
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # 测试dataset

    import torchvision
    from torchvision import datasets
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt 
    import torchvision.transforms as transforms

    batch_size = 16
    datadir = "D:\\Mydata\\Graduation_design_VAD\\Dataset\\avenue\\testing\\frames"

    train_data = VadDataset(datadir, transform=transforms.Compose([transforms.ToTensor()]), resize_height=256, resize_width=256)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    unloader = transforms.ToPILImage()

    X = next(iter(train_loader))
    print(X.shape)
