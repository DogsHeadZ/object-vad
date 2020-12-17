import torchvision.transforms as transforms
from torch.autograd import Variable
from vad_dataloader import VadDataset
import numpy as np
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import glob
import yaml
import utils
import argparse
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--name', default=None)
parser.add_argument('--tag', default=None)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

utils.set_gpu(args.gpu)

#### make datasets ####
# test
test_folder = config['dataset_path'] + config['train_dataset_type'] + "/testing/frames"

# Loading dataset
test_dataset_args = config['test_dataset_args']

test_dataset = VadDataset(test_folder, transforms.Compose([
    transforms.ToTensor(),
]), resize_height=test_dataset_args['h'], resize_width=test_dataset_args['w'], time_step=test_dataset_args['t_length'] - 1)

test_batch = DataLoader(test_dataset, batch_size=test_dataset_args['batch_size'],
                             shuffle=False, num_workers=test_dataset_args['num_workers'], drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(config['model_dir'])
model.cuda()
m_items = torch.load(config['m_items_dir'])


labels = np.load('./data/frame_labels_'+config['test_dataset_type']+'.npy')
if config['test_dataset_type'] == 'shanghai':
    labels = np.expand_dims(labels, 0)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    print(video_name)
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []

print('Evaluation of', config['test_dataset_type'])
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']

model.eval()

for k,(imgs) in enumerate(test_batch):
    if k == label_length-4*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    outputs, feas= model.forward(imgs[:,0:3*4])
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()

    psnr_list[videos_list[video_num].split('/')[-1]].append(utils.psnr(mse_imgs))

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = utils.AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', config['test_dataset_type'])
print('AUC: ', accuracy*100, '%')
