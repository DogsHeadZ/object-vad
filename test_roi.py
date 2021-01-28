import torch
import cv2
import os
import glob
import numpy as np
import scipy.io

from getROI import *
from getFlow import *

from vad_dataloader_frameobject import *

from flownet2.utils_flownet2 import flow_utils

def abnormal_bbox(ground_truth): #计算帧中的异常区域bbox,正常帧返回空

    # 计算异常区域的矩形包围框
    mask = cv2.threshold(ground_truth, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        return (x, y, x+w, y+w)
    return []

def is_normal_object(ground_truth_bbox, bbox): #判断是否是正常的物体
    if ground_truth_bbox == []:
        return True
    # elif overlap_area(ground_truth_bbox, bbox) / area(bbox) < 0.2:
    elif overlap_area(ground_truth_bbox, bbox) == 0:
        return True
    else:
        return False


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init yolov5
    yolo = attempt_load("yolov5/weights/yolov5s.pt", map_location=device)  # load FP32 model


    # video: ped2/
    dataset = "/data0/lyx/VAD_datasets/ped2/testing/frames/"
    videos = sorted(os.listdir(dataset))
    videos_ground_truth = "/data0/lyx/VAD_datasets/UCSDped2/Test/"
    video_ground_truth_list = sorted(glob.glob( os.path.join(videos_ground_truth, "*_gt") ))
    # print(video_ground_truth_list)
    
    for index in range(len(videos)):
        video = os.path.join(dataset, videos[index], "*")
        frame_list = sorted( glob.glob(video) )
        # print(frame_list)
        ground_truth_list = sorted(glob.glob(os.path.join(video_ground_truth_list[index],"*.bmp")))
        # print(ground_truth_list)
        img_size = cv2.imread(frame_list[0]).shape[0:2]

        for i in range( len(frame_list)-4 ): #滑窗长度为5
            ground_truth_bbox = abnormal_bbox( cv2.imread(ground_truth_list[i], cv2.IMREAD_GRAYSCALE) )
            flag = 0
            if ground_truth_bbox == []:
                continue

            # frame roi
            roi = RoI(frame_list[i:i+5], "ped2" , yolo, device)
            
            for bbox in roi:
                # save
                if is_normal_object(ground_truth_bbox, bbox) == False:
                    flag = 1
                    break
            if flag == 0:
                print("save ped2 video: {} frame: {}".format(videos[index], i))

