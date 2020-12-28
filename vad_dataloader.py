import numpy as np
from collections import OrderedDict
import os
import glob
import cv2

import torch.utils.data as data

def np_load_frame_roi(filename, resize_height, resize_width, bbox):

    (xmin, ymin, xmax, ymax) = bbox 
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized

class VadDataset(data.Dataset):
    def __init__(self, video_folder, bbox_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.bbox = bbox_folder
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
            bboxes_file = os.path.join(self.bbox, video_name) + '.npy'
            all_bboxes = np.load( bboxes_file, allow_pickle=True )
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):     #减掉_time_step为了刚好能够滑窗到视频尾部
                bboxes = all_bboxes[i]
                for bbox in bboxes:  
                    frames.append([self.videos[video_name]['frame'][i], bbox])          #frames存储着训练时每段视频片段的首帧文件路径和bounding box，根据首帧向后进行滑窗即可得到这段视频片段
            
        # print("frames: ",frames)           
        return frames               
            
        
    def __getitem__(self, index):
        # video_name = self.samples[index][0].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        video_name = os.path.split(os.path.split(self.samples[index][0])[0])[1]
        # print(video_name)
        # frame_name = int(self.samples[index][0].split('/')[-1].split('.')[-2])
        frame_name = int( os.path.split(self.samples[index][0])[1].split('.')[-2] )
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame_roi(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width, self.samples[index][1])   #根据首帧图片名便可加载一段视频片段
            if self.transform is not None:
                batch.append(self.transform(image))
        return np.concatenate(batch, axis=0)     #最后即返回这段视频片段大小为[（_time_step+um_pred）*图片的通道数, _resize_height, _resize_width]
        
        
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
    datadir = "../Dataset/avenue/testing/frames"

    train_data = VadDataset(datadir, "./bboxes/avenue/test", transform=transforms.Compose([transforms.ToTensor()]), resize_height=256, resize_width=256)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    unloader = transforms.ToPILImage()

    X = next(iter(train_loader))
    print(X.shape)
    for i in range(5):
        for j in range(16):
            plt.subplot(5,16,i*16+j+1)
            img = X[j,i*3:i*3+3].cpu().clone()
            img = img.squeeze(0) 
            img = unloader(img)           
            plt.imshow(img)  
    plt.show()
