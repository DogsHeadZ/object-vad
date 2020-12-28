import argparse
import os
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from collections import OrderedDict
import glob
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2

import utils
from vad_dataloader_object import VadDataset
from models.preAE import PreAE

import torchvision.transforms as transforms

def train(config):

    #### make datasets ####
    # train
    train_folder = config['dataset_path'] + config['train_dataset_type'] + "/training/frames"
    test_folder = config['dataset_path'] + config['train_dataset_type'] + "/testing/frames"

    # Loading dataset
    train_dataset_args = config['train_dataset_args']
    test_dataset_args = config['test_dataset_args']
    train_dataset = VadDataset(train_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=train_dataset_args['h'], resize_width=train_dataset_args['w'], time_step=train_dataset_args['t_length'] - 1)

    test_dataset = VadDataset(test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=test_dataset_args['h'], resize_width=test_dataset_args['w'], time_step=test_dataset_args['t_length'] - 1)

    train_dataloader = DataLoader(train_dataset, batch_size=train_dataset_args['batch_size'],
                                  shuffle=True, num_workers=train_dataset_args['num_workers'], drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_args['batch_size'],
                                 shuffle=False, num_workers=test_dataset_args['num_workers'], drop_last=False)

    for o, imgs in enumerate(tqdm(train_dataloader, desc='train', leave=False)):

        for i, channels_per_img in enumerate(imgs):
            channels_per_img = channels_per_img.cuda()[0]
            objects = [channels_per_img[3 * i: 3 * (i+1)] for i in range(channels_per_img.shape[0] // 3)]

            for j, object in enumerate(objects):

                object = object.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(
                    torch.uint8).cpu().numpy()
                # RGB转BRG
                # object = cv2.cvtColor(object, cv2.COLOR_RGB2BGR)

                cv2.imwrite('objects/' + str(i) + '_' + str(j) + '.jpg', object)

        if o == 0:
            break


if __name__ == '__main__':
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
    train(config)
