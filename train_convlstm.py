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
# from flownet2.models import FlowNet2


import utils
from vad_dataloader import VadDataset
# from models.preAE import PreAE
from models.unet import UNet
from models.networks import define_G
from models.pix2pix_networks import PixelDiscriminator
# from liteFlownet.lite_flownet import Network, batch_estimate
from losses import *

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
    labels = np.load('./data/frame_labels_' + config['test_dataset_type'] + '.npy')
    if config['test_dataset_type'] == 'shanghai':
        labels = np.expand_dims(labels, 0)
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    labels_list = []
    label_length = 0
    psnr_list = {}
    for video in sorted(videos_list):
        # video_name = video.split('/')[-1]

        # windows 
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
    num_unet_layers = 4
    discriminator_num_filters = [128, 256, 512, 512]

    # for gradient loss
    alpha = 1
    # for int loss
    l_num = 2
    pretrain = False

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
        # generator = UNet(n_channels=train_dataset_args['c']*(train_dataset_args['t_length']-1),
        #                  layer_nums=num_unet_layers, output_channel=train_dataset_args['c'])
        model = PreAE(train_dataset_args['c'], train_dataset_args['t_length'], **config['model_args'])
    else:
        raise Exception('The generator is not implemented')

    # generator = torch.load('save/avenue_cycle_generator_convlstm_flownet2_0103/generator-epoch-199.pth')


    discriminator=PixelDiscriminator(train_dataset_args['c'],discriminator_num_filters,use_norm=False)
    # discriminator = torch.load('save/avenue_cycle_generator_convlstm_flownet2_0103/discriminator-epoch-199.pth')

    # if not pretrain:
    #     generator.apply(weights_init_normal)
    #     discriminator.apply(weights_init_normal)

    # if use flownet
    # if config['flow_model'] == 'flownet2':
    #     flownet2SD_model_path = 'flownet2/FlowNet2_checkpoint.pth.tar'
    #     flow_network = FlowNet2(args).eval()
    #     flow_network.load_state_dict(torch.load(flownet2SD_model_path)['state_dict'])
    # elif config['flow_model'] == 'liteflownet':
    #     lite_flow_model_path = 'liteFlownet/network-sintel.pytorch'
    #     flow_network = Network().eval()
    #     flow_network.load_state_dict(torch.load(lite_flow_model_path))

    # different range with the source version, should change
    lam_int = 1.0 * 2
    lam_gd = 1.0 * 2
    # here we use no flow loss
    lam_op = 0  # 2.0
    lam_adv = 0.05
    adversarial_loss = Adversarial_Loss()
    discriminate_loss = Discriminate_Loss()
    gd_loss = Gradient_Loss(alpha, train_dataset_args['c'])
    op_loss = Flow_Loss()
    int_loss = Intensity_Loss(l_num)
    step = 0

    utils.log('initializing the model with Generator-Unet {} layers,'
          'PixelDiscriminator with filters {} '.format(num_unet_layers,discriminator_num_filters))

    g_lr = 0.0002
    d_lr = 0.00002
    optimizer_G = torch.optim.Adam(generator.parameters(),lr=g_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=d_lr)

    # # optimizer setting
    # params_encoder = list(generator.encoder.parameters())
    # params_decoder = list(generator.decoder.parameters())
    # params = params_encoder + params_decoder
    # optimizer, lr_scheduler = utils.make_optimizer(
    #     params, config['optimizer'], config['optimizer_args'])
    #
    # loss_func_mse = nn.MSELoss(reduction='none')

    # parallel if muti-gpus
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        # # if use flownet
        # flow_network.cuda()
        adversarial_loss.cuda()
        discriminate_loss.cuda()
        gd_loss.cuda()
        op_loss.cuda()
        int_loss.cuda()

    if config.get('_parallel'):
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        # if use flownet
        # flow_network = nn.DataParallel(flow_network)
        adversarial_loss = nn.DataParallel(adversarial_loss)
        discriminate_loss = nn.DataParallel(discriminate_loss)
        gd_loss = nn.DataParallel(gd_loss)
        op_loss = nn.DataParallel(op_loss)
        int_loss = nn.DataParallel(int_loss)

    # Training
    utils.log('Start train')
    max_accuracy = 0
    base_channel_num = train_dataset_args['c'] * (train_dataset_args['t_length'] - 1)
    save_epoch = 5 if config['save_epoch'] is None else config['save_epoch']
    for epoch in range(config['epochs']):

        generator.train()
        for j, imgs in enumerate(tqdm(train_dataloader, desc='train', leave=False)):
            imgs = imgs.cuda()
            input = imgs[:, :-1, ]
            input_last = input[:, -1, ]
            target = imgs[:, -1, ]
            # input = input.view(input.shape[0], -1, input.shape[-2],input.shape[-1])

            # only for debug
            # input0=imgs[:, 0,]
            # input1 = imgs[:, 1, ]
            # gt_flow_esti_tensor = torch.cat([input0, input1], 1)
            # flow_gt = batch_estimate(gt_flow_esti_tensor, flow_network)[0]
            # objectOutput = open('./out_train.flo', 'wb')
            # np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
            # np.array([flow_gt.size(2), flow_gt.size(1)], np.int32).tofile(objectOutput)
            # np.array(flow_gt.detach().cpu().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
            # objectOutput.close()
            # break

            # ------- update optim_G --------------
            outputs = generator(input)
            # pred_flow_tensor = torch.cat([input_last, outputs], 1)
            # gt_flow_tensor = torch.cat([input_last, target], 1)
            # flow_pred = batch_estimate(pred_flow_tensor, flow_network)
            # flow_gt = batch_estimate(gt_flow_tensor, flow_network)

            # if you want to use flownet2SD, comment out the part in front
            

            # #### if use flownet ####
            # pred_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,input.shape[-2],input.shape[-1]),
            #                                    outputs.view(-1,3,1,input.shape[-2],input.shape[-1])], 2)
            # gt_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,input.shape[-2],input.shape[-1]),
            #                                  target.view(-1,3,1,input.shape[-2],input.shape[-1])], 2)

            # flow_gt=flow_network(gt_flow_esti_tensor*255.0)
            # flow_pred=flow_network(pred_flow_esti_tensor*255.0)
            ##############################
            # g_op_loss = op_loss(flow_pred, flow_gt) ## flow loss
            g_op_loss = 0
            g_adv_loss = adversarial_loss(discriminator(outputs))
            
            g_int_loss = int_loss(outputs, target)
            g_gd_loss = gd_loss(outputs, target)
            g_loss = lam_adv * g_adv_loss + lam_gd * g_gd_loss + lam_op * g_op_loss + lam_int * g_int_loss

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            train_psnr = utils.psnr_error(outputs,target)

            # ----------- update optim_D -------
            optimizer_D.zero_grad()
            d_loss = discriminate_loss(discriminator(target), discriminator(outputs.detach()))
            d_loss.backward()
            optimizer_D.step()
            # break
        # lr_scheduler.step()

        utils.log('----------------------------------------')
        utils.log('Epoch:' + str(epoch + 1))
        utils.log('----------------------------------------')
        utils.log("g_loss: {} d_loss {}".format(g_loss, d_loss))
        utils.log('\t gd_loss {}, op_loss {}, int_loss {} ,'.format(g_gd_loss, g_op_loss, g_int_loss))
        utils.log('\t train psnr{}'.format(train_psnr))

        # Testing
        utils.log('Evaluation of ' + config['test_dataset_type'])
        for video in sorted(videos_list):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[-1]
            psnr_list[video_name] = []

        generator.eval()
        video_num = 0
        # label_length += videos[videos_list[video_num].split('/')[-1]]['length']
        label_length = videos[os.path.split(videos_list[video_num])[1]]['length']
        for k, imgs in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
            if k == label_length - 4 * (video_num + 1):
                video_num += 1
                label_length += videos[os.path.split(videos_list[video_num])[1]]['length']
            imgs = imgs.cuda()
            input = imgs[:, :-1, ]
            target = imgs[:, -1, ]
            # input = input.view(input.shape[0], -1, input.shape[-2], input.shape[-1])

            outputs = generator(input)
            mse_imgs = int_loss((outputs + 1) / 2, (target + 1) / 2).item()
            # psnr_list[videos_list[video_num].split('/')[-1]].append(utils.psnr(mse_imgs))
            psnr_list[ os.path.split(videos_list[video_num])[1] ].append(utils.psnr(mse_imgs))

        # Measuring the abnormality score and the AUC
        anomaly_score_total_list = []
        for video in sorted(videos_list):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[1]
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
            torch.save(discriminator, os.path.join(
                save_path, 'discriminator-epoch-{}.pth'.format(epoch)))

        if accuracy > max_accuracy:
            torch.save(generator, os.path.join(
                save_path, 'generator-max'))
            torch.save(discriminator, os.path.join(
                save_path, 'discriminator-max'))

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