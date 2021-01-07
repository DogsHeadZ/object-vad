import argparse
import os
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


import utils
from vad_dataloader import VadDataset
from model.preAE import PreAE


# 要删的
import torchvision.transforms as transforms
import torch.optim as optim


def train(config):
    #### set the save and log path ####
    svname = args.name
    if svname is None:
        svname = config['train_dataset_type'] + '_' + config['model']
        # svname += '_' + config['model_args']['encoder']
        # if config['model_args']['classifier'] != 'linear-classifier':
        #     svname += '-' + config['model_args']['classifier']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'classifier_config.yaml'), 'w'))

    #### make datasets ####
    # train
    train_folder = config['dataset_path'] + config['train_dataset_type'] + "/training/frames"
    test_folder = config['dataset_path'] + config['train_dataset_type'] + "/testing/frames"
    train_bboxes = "./bboxes/" + config['train_dataset_type'] + "/train"
    test_bboxes = "./bboxes/" + config['train_dataset_type'] + "/test"


    # Loading dataset
    train_dataset_args = config['train_dataset_args']
    test_dataset_args = config['test_dataset_args']
    train_dataset = VadDataset(train_folder, train_bboxes, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=train_dataset_args['h'], resize_width=train_dataset_args['w'], time_step=train_dataset_args['t_length'] - 1)

    test_dataset = VadDataset(test_folder, test_bboxes, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=test_dataset_args['h'], resize_width=test_dataset_args['w'], time_step=test_dataset_args['t_length'] - 1)

    train_dataloader = DataLoader(train_dataset, batch_size=train_dataset_args['batch_size'],
                                  shuffle=True, num_workers=train_dataset_args['num_workers'], drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_args['batch_size'],
                                 shuffle=False, num_workers=test_dataset_args['num_workers'], drop_last=False)

    # for test---- prepare labels
    labels = np.load('./data/frame_labels_' + config['test_dataset_type'] + '.npy')
    if config['test_dataset_type'] == 'shanghai':
        labels = np.expand_dims(labels, 0)
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    labels_list = []
    label_length = 0
    psnr_list = {}
    for video in sorted(videos_list):
        video_name = os.path.split(video)[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
        labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []

    # Model setting
    model = PreAE(train_dataset_args['c'], train_dataset_args['t_length'], **config['model_args'])

    # optimizer setting
    params_encoder = list(model.encoder.parameters())
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer, lr_scheduler = utils.make_optimizer(
        params, config['optimizer'], config['optimizer_args'])

    loss_func_mse = nn.MSELoss(reduction='none')

    # parallel if muti-gpus
    if torch.cuda.is_available():
        model.cuda()
    if config.get('_parallel'):
        model = nn.DataParallel(model)

    # Training
    utils.log('Start train')
    max_accuracy = 0
    base_channel_num  = train_dataset_args['c'] * (train_dataset_args['t_length'] - 1)
    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):
        model.train()
        for j, imgs in enumerate(tqdm(train_dataloader, desc='train', leave=False)):
            imgs = imgs.cuda()
            outputs, feas = model(imgs[:, 0: base_channel_num])
            optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, base_channel_num:]))
            loss = loss_pixel
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        utils.log('----------------------------------------')
        utils.log('Epoch:' + str(epoch + 1))
        utils.log('----------------------------------------')
        utils.log('Loss: Reconstruction {:.6f}'.format(loss_pixel.item()))

        # Testing
        utils.log('Evaluation of ' + config['test_dataset_type'])
        for video in sorted(videos_list):
            video_name = os.path.split(video)[-1]
            psnr_list[video_name] = []

        # print(psnr_list.keys())

        model.eval()
        video_num = 0
        label_length = videos[os.path.split(videos_list[video_num])[-1]]['length']
        bboxes_list = sorted(os.listdir(test_bboxes))
        # print(bboxes_list)
        bboxes = np.load(os.path.join(test_bboxes,bboxes_list[0]), allow_pickle=True) 

        mses = []
        frame = 0
        total_frame = 0
        for k, imgs in enumerate(tqdm(test_dataloader, desc='test', leave=True)):
            if total_frame == label_length - 4 * (video_num + 1):
                video_num += 1
                label_length += videos[os.path.split(videos_list[video_num])[-1]]['length']
                bboxes = np.load(os.path.join(test_bboxes,bboxes_list[video_num]), allow_pickle=True) 
                frame = 0
            imgs = imgs.cuda()
            outputs, feas = model.forward(imgs[:, 0: base_channel_num])
            mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, base_channel_num:] + 1) / 2)).item()
            
            mses.append(mse_imgs)
            
            if len(mses) == len(bboxes[frame]):
                psnr_list[os.path.split(videos_list[video_num])[-1]].append(utils.psnr(max(mses)))
                frame += 1
                total_frame += 1
                mses = []

        # Measuring the abnormality score and the AUC
        anomaly_score_total_list = []
        for video in sorted(videos_list):
            video_name = os.path.split(video)[-1]
            anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])

        anomaly_score_total_list = np.asarray(anomaly_score_total_list)
        accuracy = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

        utils.log('The result of ' + config['test_dataset_type'])
        utils.log('AUC: ' + str(accuracy * 100) + '%')

        # Save the model
        if epoch % save_epoch == 0 or epoch == config['epochs'] - 1:
            torch.save(model, os.path.join(
                save_path, 'model-epoch-{}.pth'.format(epoch)))

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model, os.path.join(save_path, 'max-va-model.pth'))

        utils.log('----------------------------------------')

    utils.log('Training is finished')


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
