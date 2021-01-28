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


import utils
from vad_dataloader import VadDataset
from vad_dataloader_frameflow import VadDataset

from models.preAE import PreAE
from models.unet import UNet
from models.networks import define_G
from models.pix2pix_networks import PixelDiscriminator


from losses import *

import torchvision.transforms as transforms

import cv2
import scipy.io

import matplotlib.pyplot as plt

def roi_mse(output, target, loss_fun, frame_bboxes, height_coef, width_coef):
    mse = []

    for bbox in frame_bboxes:
        (xmin, ymin, xmax, ymax) = bbox
        xmin = int(xmin*width_coef)
        xmax = int(xmax*width_coef)
        ymin = int(ymin*height_coef)
        ymax = int(ymax*height_coef)

        output_roi = output[:, :, ymin:ymax, xmin:xmax]
        target_roi = target[:, :, ymin:ymax, xmin:xmax]

        mse.append( loss_fun(output_roi, target_roi).item() )

    return max(mse)

def draw_bbox(img, bboxes, height_coef, width_coef):
    for box in bboxes:
        (xmin, ymin, xmax, ymax) = box 
        xmin = int(xmin*width_coef)
        xmax = int(xmax*width_coef)
        ymin = int(ymin*height_coef)
        ymax = int(ymax*height_coef)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), [0,0,255] ,1)
    return img

def evaluate(test_dataloader, generator, labels_list, videos, loss_function, dataset, test_bboxes, frame_height=256, frame_width=256,
            is_visual=False, mask_labels_path=None, save_path=None, labels_dict=None):    
    #init
    psnr_list = {}
    roi_psnr_list = {}
    for key in videos.keys():
        psnr_list[key] = []
        roi_psnr_list[key] = []


    video_num = 0
    frame_index = 0
    label_length = videos[sorted(videos.keys())[video_num]]['length']
    bboxes_list = sorted(os.listdir(test_bboxes))
    bboxes = np.load(os.path.join(test_bboxes,bboxes_list[0]), allow_pickle=True) 

    WIDTH, HEIGHT = 640,360
    # 保存可视化结果
    if is_visual:
        # 异常标记信息
        mask_labels = sorted(glob.glob(os.path.join(mask_labels_path,"*")))
        mask_label_list = np.load(mask_labels[0], allow_pickle=True)
        #TODO
        if dataset == 'avenue':
            WIDTH = 640
            HEIGHT = 360
        elif dataset == 'ped2':
            HEIGHT = 240
            WIDTH = 360
        else:
            raise ValueError("no dataset")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists("{}/video".format(save_path)):
            os.makedirs("{}/video".format(save_path))
        if not os.path.exists("{}/psnr".format(save_path)):
            os.makedirs("{}/psnr".format(save_path))

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter("{}/video/{}_01.avi".format(save_path, dataset), fourcc, 25.0, (frame_width*3+20,frame_height))


    # test
    generator.eval()
    for k, imgs in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
        if k == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[sorted(videos.keys())[video_num]]['length']
            frame_index = 0
            bboxes = np.load(os.path.join(test_bboxes,bboxes_list[video_num] ), allow_pickle=True) 

            if is_visual == True:
                out = cv2.VideoWriter("{}/video/{}_{}.avi".format(save_path, dataset, sorted(videos.keys())[video_num]), fourcc, 25.0, (frame_width*3+20,frame_height))
                mask_label_list = np.load(mask_labels[video_num], allow_pickle=True)
        imgs = imgs.cuda()
        input = imgs[:, :-1, ]
        target = imgs[:, -1, ]

        # print(input.data.shape)

        outputs = generator(input)  #[c, h, w]
        # print(outputs.data.shape)

        # mse roi
        roi_mse_imgs = roi_mse(outputs, target, loss_function, bboxes[frame_index], frame_height/HEIGHT, frame_width/WIDTH)
        roi_psnr_list[ sorted(videos.keys())[video_num] ].append(utils.psnr(roi_mse_imgs))
        
        # mse frame        
        mse_imgs = loss_function(outputs, target).item() 
        psnr_list[ sorted(videos.keys())[video_num] ].append(utils.psnr(mse_imgs))
        

        if is_visual:
            ################ show predict frame ######################
            real_frame = target.squeeze().data.cpu().numpy().transpose(1,2,0)
            predict_frame = outputs.squeeze().data.cpu().numpy().transpose(1,2,0)
            # diff = cv2.absdiff(real_frame, predict_frame)
            diff = (real_frame-predict_frame)**2
            diff = np.uint8((diff[:,:,0] + diff[:,:,1] + diff[:,:,2])*255.0)
            diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
            diff = draw_bbox(diff, bboxes[frame_index], 255.0/HEIGHT, 255.0/WIDTH)

            real_frame = np.uint8(real_frame*255.0)
            predict_frame = np.uint8(predict_frame*255.0)

            # add mask label to real img
            mask = np.uint8(mask_label_list[frame_index]*255.0)
            mask = cv2.resize(mask, (256,256) )
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask[:,:,1] = np.zeros([256,256])
            mask[:,:,0] = np.zeros([256,256])
            real_frame = cv2.addWeighted(real_frame,1, mask,0.6, 0)

            compare = np.concatenate([real_frame, np.zeros([256,20,3]), predict_frame, diff], axis=1 )
            # cv2.imshow("real_frame", real_frame)
            # cv2.imshow("diff", diff)
            # cv2.imshow("compare", np.uint8(compare) )
            # cv2.waitKey(1)
            compare = np.uint8(compare)
            # add text
            cv2.putText(compare, "video: {}, frame:{}, psnr: {}".format(video_num+1, frame_index, utils.psnr(mse_imgs)), (256*2+20, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (200,255,255), 1 )
            # print("putText")
            out.write(compare)

        frame_index += 1    
    
    # Measuring the abnormality score and the AUC
    # 这个地方想了一下应该还是全局做归一化
    anomaly_score_total_list = []
    roi_anomaly_score_total_list = []
    for video_name in sorted(videos.keys()):
        # anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])
        # roi_anomaly_score_total_list += utils.anomaly_score_list(roi_psnr_list[video_name])
        anomaly_score_total_list += psnr_list[video_name]
        roi_anomaly_score_total_list += roi_psnr_list[video_name]

        if is_visual:
            plt.figure(figsize=(10,2))
            # 绘制psnr
            plt.plot(psnr_list[video_name])
            plt.xlabel('t')
            plt.ylabel('psnr')

            min_ = min(list(psnr_list[video_name]))
            max_ = max(list(psnr_list[video_name]))

            # 绘制真值
            plt.fill_between(np.linspace(0, len(labels_dict[video_name][0]), len(labels_dict[video_name][0])), 
                np.array(min_), (max_-min_)*labels_dict[video_name][0]+min_, facecolor='r', alpha=0.3)
            plt.savefig("{}/psnr/{}_{}_frame_psnr.jpg".format(save_path, dataset, video_name))
            print(video_name)

            plt.figure(figsize=(10,2))
            # 绘制psnr
            plt.plot(roi_psnr_list[video_name])
            plt.xlabel('t')
            plt.ylabel('psnr')

            min_ = min(list(roi_psnr_list[video_name]))
            max_ = max(list(roi_psnr_list[video_name]))

            # 绘制真值
            plt.fill_between(np.linspace(0, len(labels_dict[video_name][0]), len(labels_dict[video_name][0])), 
                np.array(min_), (max_-min_)*labels_dict[video_name][0]+min_, facecolor='r', alpha=0.3)
            plt.savefig("{}/psnr/{}_{}_roi_psnr.jpg".format(save_path, dataset, video_name))

    anomaly_score_total_list = utils.anomaly_score_list(anomaly_score_total_list)
    roi_anomaly_score_total_list = utils.anomaly_score_list(roi_anomaly_score_total_list)

    # TODO
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)    
    frame_AUC = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

    roi_anomaly_score_total_list = np.asarray(roi_anomaly_score_total_list)
    roi_AUC = utils.AUC(roi_anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    # print('AUC: ' + str(accuracy * 100) + '%')


    return frame_AUC, roi_AUC


def ObjectLoss_evaluate(test_dataloader, generator, labels_list, videos, dataset, device, frame_height=256, frame_width=256,
            is_visual=False, mask_labels_path=None, save_path=None, labels_dict=None): 
    #init
    psnr_list = {}
    roi_psnr_list = {}
    for key in videos.keys():
        psnr_list[key] = []
        roi_psnr_list[key] = []

    object_loss = ObjectLoss(device, 2)

    video_num = 0
    frame_index = 0
    label_length = videos[sorted(videos.keys())[video_num]]['length']
    # bboxes_list = sorted(os.listdir(test_bboxes))
    # bboxes = np.load(os.path.join(test_bboxes,bboxes_list[0]), allow_pickle=True) 

    WIDTH, HEIGHT = 640,360
    # 保存可视化结果
    if is_visual:
        # 异常标记信息
        mask_labels = sorted(glob.glob(os.path.join(mask_labels_path,"*")))
        mask_label_list = np.load(mask_labels[0], allow_pickle=True)
        #TODO
        if dataset == 'avenue':
            WIDTH = 640
            HEIGHT = 360
        elif dataset == 'ped2':
            HEIGHT = 240
            WIDTH = 360
        else:
            raise ValueError("no dataset")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists("{}/video".format(save_path)):
            os.makedirs("{}/video".format(save_path))
        if not os.path.exists("{}/psnr".format(save_path)):
            os.makedirs("{}/psnr".format(save_path))

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter("{}/video/{}_01.avi".format(save_path, dataset), fourcc, 25.0, (frame_width*3+20,frame_height))


    # test
    generator.eval()
    for k, (imgs, bboxes, flow) in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
        if k == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[sorted(videos.keys())[video_num]]['length']
            frame_index = 0
            # bboxes = np.load(os.path.join(test_bboxes,bboxes_list[video_num] ), allow_pickle=True) 

            if is_visual == True:
                out = cv2.VideoWriter("{}/video/{}_{}.avi".format(save_path, dataset, sorted(videos.keys())[video_num]), fourcc, 25.0, (frame_width*3+20,frame_height))
                mask_label_list = np.load(mask_labels[video_num], allow_pickle=True)
        imgs = imgs.cuda()
        flow = flow.cuda()
        input = imgs[:, :-1, ]
        target = imgs[:, -1, ]

        # print(input.data.shape)

        outputs = generator(input)  #[c, h, w]
        # print(outputs.data.shape)

        mse_imgs = object_loss((outputs + 1) / 2, (target + 1) / 2, flow, bboxes).item()
        psnr_list[ sorted(videos.keys())[video_num] ].append(utils.psnr(mse_imgs))
        

        if is_visual:
            ################ show predict frame ######################
            real_frame = target.squeeze().data.cpu().numpy().transpose(1,2,0)
            predict_frame = outputs.squeeze().data.cpu().numpy().transpose(1,2,0)
            # diff = cv2.absdiff(real_frame, predict_frame)
            diff = (real_frame-predict_frame)**2
            diff = np.uint8((diff[:,:,0] + diff[:,:,1] + diff[:,:,2])*255.0)
            diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
            # diff = draw_bbox(diff, bboxes[frame_index], 255.0/HEIGHT, 255.0/WIDTH)

            real_frame = np.uint8(real_frame*255.0)
            predict_frame = np.uint8(predict_frame*255.0)

            # add mask label to real img
            mask = np.uint8(mask_label_list[frame_index]*255.0)
            mask = cv2.resize(mask, (256,256) )
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask[:,:,1] = np.zeros([256,256])
            mask[:,:,0] = np.zeros([256,256])
            real_frame = cv2.addWeighted(real_frame,1, mask,0.6, 0)

            compare = np.concatenate([real_frame, np.zeros([256,20,3]), predict_frame, diff], axis=1 )
            # cv2.imshow("real_frame", real_frame)
            # cv2.imshow("diff", diff)
            # cv2.imshow("compare", np.uint8(compare) )
            # cv2.waitKey(1)
            compare = np.uint8(compare)
            # add text
            cv2.putText(compare, "video: {}, frame:{}, psnr: {}".format(video_num+1, frame_index, utils.psnr(mse_imgs)), (256*2+20, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (200,255,255), 1 )
            # print("putText")
            out.write(compare)

        frame_index += 1    
    
    # Measuring the abnormality score and the AUC
    # 这个地方想了一下应该还是全局做归一化
    anomaly_score_total_list = []
    for video_name in sorted(videos.keys()):
        # anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])
        # roi_anomaly_score_total_list += utils.anomaly_score_list(roi_psnr_list[video_name])
        anomaly_score_total_list += psnr_list[video_name]

        if is_visual:
            plt.figure(figsize=(10,2))
            # 绘制psnr
            plt.plot(psnr_list[video_name])
            plt.xlabel('t')
            plt.ylabel('psnr')

            min_ = min(list(psnr_list[video_name]))
            max_ = max(list(psnr_list[video_name]))

            # 绘制真值
            plt.fill_between(np.linspace(0, len(labels_dict[video_name][0]), len(labels_dict[video_name][0])), 
                np.array(min_), (max_-min_)*labels_dict[video_name][0]+min_, facecolor='r', alpha=0.3)
            plt.savefig("{}/psnr/{}_{}_frame_psnr.jpg".format(save_path, dataset, video_name))
            print(video_name)

            plt.figure(figsize=(10,2))

    anomaly_score_total_list = utils.anomaly_score_list(anomaly_score_total_list)

    # TODO
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)    
    frame_AUC = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

    return frame_AUC




if __name__ == "__main__":
    
    # test data:
    test_folder = "/data0/lyx/VAD_datasets/avenue/testing/frames"
    test_dataset = VadDataset(test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=256 , resize_width = 256, time_step = 4)

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False, num_workers=0, drop_last=False)

    test_bboxes = "./bboxes/avenue/test"

    model = PreAE()
    # model_init
    weight_path = "./save/avenue_unet/models/max-frame_auc-model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generator.load_state_dict(torch.load(weight_path))
    generator = PreAE()
    weight = torch.load(weight_path)
    generator.load_state_dict(weight)
    generator.to(device)
    # print(generator)

    # loss function
    int_loss = Intensity_Loss(2)

    # labels for test
    # labels = np.load("./data/frame_labels_avenue.npy")
    labels = scipy.io.loadmat('./data/avenue_frame_labels.mat')

     # 像素级标签
    mask_labels_path = "./data/avenue_mask_labels"
    mask_labels = sorted(glob.glob(mask_labels_path))

   

    # # save video for further research
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('./convlstm_video/avenue_01.avi', fourcc, 20.0, (256*3+20,256))

    # init
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
        labels_list = np.append(labels_list, labels[video_name][0][4:])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []

    # test_dataloader, generator, labels_list, videos, loss_function, dataset, test_bboxes
    # is_visual=False, mask_labels_path=None, save_path=None, labels_dict=None):
    frame_AUC, roi_AUC = evaluate(test_dataloader, generator, labels_list, videos, int_loss, "avenue", test_bboxes,
    is_visual=True, mask_labels_path = mask_labels_path, save_path = "./video", labels_dict=labels) 
    print(frame_AUC, roi_AUC)
    
    # video_num = 0
    # frame_index = 0
    # label_length = videos[os.path.split(videos_list[video_num])[-1]]['length']
    # bboxes_list = sorted(os.listdir(test_bboxes))
    # # print(bboxes_list)
    # bboxes = np.load(os.path.join(test_bboxes,bboxes_list[0]), allow_pickle=True) 
    # # print(bboxes)
    # mask_label_list = scipy.io.loadmat(mask_labels[0])['volLabel'][0]

    # # test
    # generator.eval()
    # for k, imgs in enumerate(tqdm(test_dataloader, desc='test', leave=False)):
    #     if k == label_length - 4 * (video_num + 1):
    #         video_num += 1
    #         label_length += videos[os.path.split(videos_list[video_num])[1]]['length']
    #         frame_index = 0
    #         out = cv2.VideoWriter('./convlstm_video/avenue_{}.avi'.format(os.path.split(videos_list[video_num])[1]), fourcc, 20.0, (256*3+20,256))
    #         bboxes = np.load(os.path.join(test_bboxes,bboxes_list[video_num] ), allow_pickle=True) 
    #         mask_label_list = scipy.io.loadmat(mask_labels[video_num])['volLabel'][0]
        
    #     imgs = imgs.cuda()
    #     input = imgs[:, :-1, ]
    #     target = imgs[:, -1, ]

    #     # print(input.data.shape)

    #     outputs = generator(input)  #[c, h, w]
    #     # print(outputs.data.shape)

    #     # mse roi
    #     mse_imgs = roi_mse(outputs, target, bboxes[frame_index], 255.0/360.0, 255.0/640.0)
    #     # mse frame
    #     # mse_imgs = int_loss((outputs + 1) / 2, (target + 1) / 2).item() 
    #     # psnr_list[videos_list[video_num].split('/')[-1]].append(utils.psnr(mse_imgs))
    #     psnr_list[ os.path.split(videos_list[video_num])[1] ].append(utils.psnr(mse_imgs))

    #     ################ show predict frame ######################
    #     real_frame = target.squeeze().data.cpu().numpy().transpose(1,2,0)
    #     predict_frame = outputs.squeeze().data.cpu().numpy().transpose(1,2,0)
    #     # diff = cv2.absdiff(real_frame, predict_frame)
    #     diff = (real_frame-predict_frame)**2
    #     diff = np.uint8((diff[:,:,0] + diff[:,:,1] + diff[:,:,2])*255.0)
    #     diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
    #     # diff = draw_bbox(diff, bboxes[frame_index], 255.0/360.0, 255.0/640.0)

    #     real_frame = np.uint8(real_frame*255.0)
    #     predict_frame = np.uint8(predict_frame*255.0)

    #     # add mask label to real img
    #     mask = np.uint8(mask_label_list[frame_index]*255.0)
    #     mask = cv2.resize(mask, (256,256) )
    #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #     mask[:,:,1] = np.zeros([256,256])
    #     mask[:,:,0] = np.zeros([256,256])
    #     # real_frame = cv2.addWeighted(real_frame,1, mask,0.6, 0)

    #     compare = np.concatenate([real_frame, np.zeros([256,20,3]), predict_frame, diff], axis=1 )
    #     # cv2.imshow("real_frame", real_frame)
    #     # cv2.imshow("diff", diff)
    #     # cv2.imshow("compare", np.uint8(compare) )
    #     # cv2.waitKey(1)
    #     compare = np.uint8(compare)
    #     # add text
    #     # cv2.putText(compare, "video: {}, frame:{}, psnr: {}".format(video_num+1, frame_index, utils.psnr(mse_imgs)), (256*2+20, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (200,255,255), 1 )
    #     # print("putText")
    #     out.write(compare)

    #     frame_index += 1
    #     # break

    # np.save("./convlstm_video/avenue_psnr_object_lyx.npy", psnr_list)

    # # Measuring the abnormality score and the AUC
    # anomaly_score_total_list = []
    # for video in sorted(videos_list):
    #     # video_name = video.split('/')[-1]
    #     video_name = os.path.split(video)[1]
    #     anomaly_score_total_list += utils.anomaly_score_list(psnr_list[video_name])

    # anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    # accuracy = utils.AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    # print('AUC: ' + str(accuracy * 100) + '%')

    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()