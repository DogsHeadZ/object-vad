import scipy.io as scio
import numpy as np
from collections import OrderedDict
import glob
import os
# dataFile = 'avenue.mat'
# data = scio.loadmat(dataFile)
# print(data['gt'][0][0])

dataFile = 'avenue.mat'
data = scio.loadmat(dataFile)
print(data)

labels = np.load('./data/frame_labels_' + 'avenue' + '.npy')
test_folder = '/Users/feihu/data/video/avenue/testing/frames'
videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
labels_list = []
label_length = 0
psnr_list = {}
i = 0
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    i = i + 1
    if i == 18:
        print(label_length)
    if i == 19 :
        print(label_length)
print(labels[0][14727: 14975])
print(len(labels[0]))