import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import time
from tqdm import tqdm
from collections import OrderedDict
import glob
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable


import utils
from vad_dataloader import VadDataset
from models.preAE import PreAE

import torchvision.transforms as transforms

from evaluate import *


def train(config):
    #### set the save and log path ####
    # svname = args.name
    # if svname is None:
    #     svname = config['train_dataset_type'] + '_' + config['model']
    #     # svname += '_' + config['model_args']['encoder']
    #     # if config['model_args']['classifier'] != 'linear-classifier':
    #     #     svname += '-' + config['model_args']['classifier']
    # if args.tag is not None:
    #     svname += '_' + args.tag
    # save_path = os.path.join('./save', svname)
    save_path = config['save_path']
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['save_path'], 'tensorboard'))
    yaml.dump(config, open(os.path.join(config['save_path'], 'classifier_config.yaml'), 'w'))

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

    # for test---- prepare labels
    labels = scipy.io.loadmat(config['label_path'])
    if config['test_dataset_type'] == 'shanghai':
        labels = np.expand_dims(labels, 0)
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    labels_list = []
    label_length = 0
    psnr_list = {}
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
        labels_list = np.append(labels_list, labels[video_name][0][4:])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []

    # Model setting
    if config['generator'] == 'cycle_generator_convlstm':
        ngf = 64
        netG = 'resnet_6blocks'
        norm = 'instance'
        no_dropout = False
        init_type = 'normal'
        init_gain = 0.02
        gpu_ids = []
        model = define_G(train_dataset_args['c'], train_dataset_args['c'],
                             ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)
    elif config['generator'] == 'unet':
        # TODO
        model = UNet(n_channels=train_dataset_args['c']*(train_dataset_args['t_length']-1),
                         layer_nums=num_unet_layers, output_channel=train_dataset_args['c'])
    else:
        raise Exception('The generator is not implemented')

    # optimizer setting
    params_encoder = list(model.parameters())
    params_decoder = list(model.parameters())
    params = params_encoder + params_decoder
    optimizer, lr_scheduler = utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    # set loss, different range with the source version, should change
    lam_int = 1.0 * 2
    lam_gd = 1.0 * 2
    # TODO here we use no flow loss
    # lam_op = 0  # 2.0
    # op_loss = Flow_Loss()
    
    adversarial_loss = Adversarial_Loss()
    # # TODO if use adv
    # lam_adv = 0.05
    # discriminate_loss = Discriminate_Loss()
    alpha = 1
    l_num = 2
    gd_loss = Gradient_Loss(alpha, train_dataset_args['c'])    
    int_loss = Intensity_Loss(l_num)

    # parallel if muti-gpus
    if torch.cuda.is_available():
        model.cuda()
    if config.get('_parallel'):
        model = nn.DataParallel(model)

    # Training
    utils.log('Start train')
    max_frame_AUC, max_roi_AUC = 0,0
    base_channel_num  = train_dataset_args['c'] * (train_dataset_args['t_length'] - 1)
    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):

        model.train()
        for j, imgs in enumerate(tqdm(train_dataloader, desc='train', leave=False)):
            imgs = imgs.cuda()
            # input = imgs[:, :-1, ].view(imgs.shape[0], -1, imgs.shape[-2], imgs.shape[-1])
            input = imgs[:, :-1, ]
            target = imgs[:, -1, ]
            outputs = model(input)
            optimizer.zero_grad()

            g_int_loss = int_loss(outputs, target)
            g_gd_loss = gd_loss(outputs, target)
            loss = lam_gd * g_gd_loss + lam_int * g_int_loss

            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        utils.log('----------------------------------------')
        utils.log('Epoch:' + str(epoch + 1))
        utils.log('----------------------------------------')
        utils.log('Loss: Reconstruction {:.6f}'.format(loss.item()))

        # Testing
        utils.log('Evaluation of ' + config['test_dataset_type'])   

        # Save the model
        if epoch % save_epoch == 0 or epoch == config['epochs'] - 1:
            if not os.path.exists(save_path):
                os.makedirs(save_path) 
            if not os.path.exists(os.path.join(save_path, "models")):
                os.makedirs(os.path.join(save_path, "models")) 
            # TODO
            frame_AUC, roi_AUC = evaluate(test_dataloader, model, labels_list, videos, int_loss, config['test_dataset_type'], test_bboxes=config['test_bboxes'],
                frame_height = train_dataset_args['h'], frame_width=train_dataset_args['w'], 
                is_visual=False, mask_labels_path = config['mask_labels_path'], save_path = os.path.join(save_path, "./final"), labels_dict=labels) 
            
            torch.save(model.state_dict(), os.path.join(save_path, 'models/model-epoch-{}.pth'.format(epoch)))
        else:
            frame_AUC, roi_AUC = evaluate(test_dataloader, model, labels_list, videos, int_loss, config['test_dataset_type'], test_bboxes=config['test_bboxes'],
                frame_height = train_dataset_args['h'], frame_width=train_dataset_args['w']) 

        utils.log('The result of ' + config['test_dataset_type'])
        utils.log("AUC: {}%, roi AUC: {}%".format(frame_AUC*100, roi_AUC*100))

        if frame_AUC > max_frame_AUC:
            max_frame_AUC = frame_AUC
            # TODO
            torch.save(model.state_dict(), os.path.join(save_path, 'models/max-frame_auc-model.pth'))
            evaluate(test_dataloader, model, labels_list, videos, int_loss, config['test_dataset_type'], test_bboxes=config['test_bboxes'],
                frame_height = train_dataset_args['h'], frame_width=train_dataset_args['w'], 
                is_visual=True, mask_labels_path = config['mask_labels_path'], save_path = os.path.join(save_path, "./frame_best"), labels_dict=labels) 
        if roi_AUC > max_roi_AUC:
            max_roi_AUC = roi_AUC
            torch.save(model.state_dict(), os.path.join(save_path, 'models/max-roi_auc-model.pth'))
            evaluate(test_dataloader, model, labels_list, videos, int_loss, config['test_dataset_type'], test_bboxes=config['test_bboxes'],
                frame_height = train_dataset_args['h'], frame_width=train_dataset_args['w'], 
                is_visual=True, mask_labels_path = config['mask_labels_path'], save_path = os.path.join(save_path, "./roi_best"), labels_dict=labels) 

        utils.log('----------------------------------------')

    utils.log('Training is finished')
    utils.log('max_frame_AUC: {}, max_roi_AUC: {}'.format(max_frame_AUC, max_roi_AUC))


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