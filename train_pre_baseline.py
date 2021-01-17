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
from flownet2.models import FlowNet2


import utils
from vadmodels.preAE import PreAE
from vadmodels.unet import UNet
from vadmodels.networks import define_G
from vadmodels.pix2pix_networks import PixelDiscriminator
from liteFlownet.lite_flownet import Network, batch_estimate
from losses import *
from vad_dataloader_objectflow import VadDataset

import torchvision.transforms as transforms

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(config):
    #### set the save and log path ####
    svname = args.name
    if svname is None:
        svname = config['train_dataset_type'] + '_' + config['generator'] + '_' + config['flow_model']

    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'classifier_config.yaml'), 'w'))

    device = torch.device('cuda:' + args.gpu)

    #### make datasets ####
    # train
    train_folder = config['dataset_path'] + config['train_dataset_type'] + "/training/frames"
    test_folder = config['dataset_path'] + config['train_dataset_type'] + "/testing/frames"

    # Loading dataset
    train_dataset_args = config['train_dataset_args']
    test_dataset_args = config['test_dataset_args']
    train_dataset = VadDataset(args, train_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=train_dataset_args['h'], resize_width=train_dataset_args['w'],
                               dataset=config['train_dataset_type'], time_step=train_dataset_args['t_length'] - 1,
                               device=device)

    test_dataset = VadDataset(args, test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=test_dataset_args['h'], resize_width=test_dataset_args['w'],
                              dataset=config['train_dataset_type'], time_step=test_dataset_args['t_length'] - 1,
                              device=device)

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
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
        labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
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
        generator = define_G(train_dataset_args['c'], train_dataset_args['c'],
                             ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)
    elif config['generator'] == 'unet':
        num_unet_layers = 4
        generator = UNet(n_channels=train_dataset_args['c']*(train_dataset_args['t_length']-1),
                         layer_nums=num_unet_layers, output_channel=train_dataset_args['c'])
    else:
        raise Exception('The generator is not implemented')
    # generator = torch.load('save/avenue_cycle_generator_convlstm_flownet2_0103/generator-epoch-199.pth')

    # if not pretrain:
    #     generator.apply(weights_init_normal)
    #     discriminator.apply(weights_init_normal)

    if config['flow_model'] == 'flownet2':
        flownet2SD_model_path = 'flownet2/FlowNet2_checkpoint.pth.tar'
        flow_network = FlowNet2(args).eval()
        flow_network.load_state_dict(torch.load(flownet2SD_model_path)['state_dict'])
    elif config['flow_model'] == 'liteflownet':
        lite_flow_model_path = 'liteFlownet/network-sintel.pytorch'
        flow_network = Network().eval()
        flow_network.load_state_dict(torch.load(lite_flow_model_path))

    # different range with the source version, should change
    lam_object, lam_int, lam_gd, lam_op = float(config['lam_object']), float(config['lam_int']), float(config['lam_gd']), float(config['lam_op'])

    # for gradient loss
    alpha = 1
    # for int loss
    l_num = 2
    pretrain = False
    object_loss = ObjectLoss(device, l_num)
    gd_loss = Gradient_Loss(alpha, train_dataset_args['c'])
    op_loss = Flow_Loss()
    int_loss = Intensity_Loss(l_num)

    optimizer_G, lr_scheduler = utils.make_optimizer(
            generator.parameters(), config['optimizer_G'], config['optimizer_G_args'])

    # parallel if muti-gpus
    if torch.cuda.is_available():
        generator.cuda()
        # discriminator.cuda()
        flow_network.cuda()
        object_loss.cuda()
        gd_loss.cuda()
        op_loss.cuda()
        int_loss.cuda()

    if config.get('_parallel'):
        generator = nn.DataParallel(generator)
        # discriminator = nn.DataParallel(discriminator)
        flow_network = nn.DataParallel(flow_network)
        object_loss = nn.DataParallel(object_loss)
        gd_loss = nn.DataParallel(gd_loss)
        op_loss = nn.DataParallel(op_loss)
        int_loss = nn.DataParallel(int_loss)

    # Training
    utils.log('Start train')
    max_accuracy = 0
    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):

        generator.train()
        for j, (imgs, bboxes) in enumerate(tqdm(train_dataloader, desc='train', leave=False)):

            imgs = imgs.cuda()
            input = imgs[:, :-1, ]
            input_last = input[:, -1, ]
            target = imgs[:, -1, ]

            # ------- update optim_G --------------
            outputs = generator(input)

            pred_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,input.shape[-2],input.shape[-1]),
                                               outputs.view(-1,3,1,input.shape[-2],input.shape[-1])], 2)
            gt_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,input.shape[-2],input.shape[-1]),
                                             target.view(-1,3,1,input.shape[-2],input.shape[-1])], 2)

            flow_gt=flow_network(gt_flow_esti_tensor*255.0)
            flow_pred=flow_network(pred_flow_esti_tensor*255.0)

            g_object_loss = object_loss(outputs, target, flow_gt, bboxes)
            g_int_loss = int_loss(outputs, target)
            g_gd_loss = gd_loss(outputs, target)
            g_op_loss = op_loss(flow_pred, flow_gt)

            g_loss = lam_object * g_object_loss + lam_int * g_int_loss + lam_gd * g_gd_loss + lam_op * g_op_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            train_psnr = utils.psnr_error(outputs,target)

        if lr_scheduler is not None:
            lr_scheduler.step()

        utils.log('----------------------------------------')
        utils.log('Epoch:' + str(epoch + 1))
        utils.log('----------------------------------------')
        utils.log("g_loss: {}".format(g_loss))
        utils.log('\t object_loss{}, int_loss {}, gd_loss {}, op_loss {},'.format(g_object_loss, g_int_loss, g_gd_loss, g_op_loss))
        utils.log('\t train psnr{}'.format(train_psnr))

        # Testing
        utils.log('Evaluation of ' + config['test_dataset_type'])
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            psnr_list[video_name] = []

        generator.eval()
        video_num = 0
        label_length = videos[videos_list[video_num].split('/')[-1]]['length']
        for k, (imgs, bboxes) in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
            if k == label_length - 4 * (video_num + 1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            imgs = imgs.cuda()
            input = imgs[:, :-1, ]
            input_last = input[:, -1, ]
            target = imgs[:, -1, ]

            outputs = generator(input)

            gt_flow_esti_tensor = torch.cat([input_last.view(-1, 3, 1, input.shape[-2], input.shape[-1]),
                                             target.view(-1, 3, 1, input.shape[-2], input.shape[-1])], 2)
            flow_gt = flow_network(gt_flow_esti_tensor * 255.0)

            mse_imgs = object_loss((outputs + 1) / 2, (target + 1) / 2, flow_gt, bboxes).item()
            psnr_list[videos_list[video_num].split('/')[-1]].append(utils.psnr(mse_imgs))

        # Measuring the abnormality score and the AUC
        anomaly_score_total_list = []
        for video in sorted(videos_list):
            video_name = video.split('/')[-1]
            anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])

        anomaly_score_total_list = np.asarray(anomaly_score_total_list)
        accuracy = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

        utils.log('The result of ' + config['test_dataset_type'])
        utils.log('AUC: ' + str(accuracy * 100) + '%')

        # Save the model
        if epoch % save_epoch == 0 or epoch == config['epochs'] - 1:
            # torch.save(model, os.path.join(
            #     save_path, 'model-epoch-{}.pth'.format(epoch)))

            torch.save(generator, os.path.join(
                save_path, 'generator-epoch-{}.pth'.format(epoch)))

        if accuracy > max_accuracy:
            torch.save(generator, os.path.join(
                save_path, 'generator-max'))


        utils.log('----------------------------------------')

    utils.log('Training is finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    else:
        torch.cuda.set_device(int(args.gpu))
    utils.set_gpu(args.gpu)
    train(config)
