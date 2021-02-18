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
    # elif overlap_area(ground_truth_bbox, bbox) / area(bbox) < 0.5:
    elif overlap_area(ground_truth_bbox, bbox) == 0:
        return True
    else:
        return False


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init flownet
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()
    flownet = FlowNet2(args).to(device)
    # load the state_dict
    dict_ = torch.load("flownet2/FlowNet2_checkpoint.pth.tar")
    flownet.load_state_dict(dict_["state_dict"])

    # init yolov5
    yolo = attempt_load("yolov5/weights/yolov5s.pt", map_location=device)  # load FP32 model


    # # video: ped2/
    # save_dir = "./show_object/ped2"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    #     os.mkdir(os.path.join(save_dir, "normal"))
    #     os.mkdir(os.path.join(save_dir, "abnormal"))

    # dataset = "/data0/lyx/VAD_datasets/ped2/testing/frames/"
    # videos = sorted(os.listdir(dataset))
    # videos_ground_truth = "/data0/lyx/VAD_datasets/UCSDped2/Test/"
    # video_ground_truth_list = sorted(glob.glob( os.path.join(videos_ground_truth, "*_gt") ))
    # # print(video_ground_truth_list)
    
    # for index in range(len(videos)):
    #     video = os.path.join(dataset, videos[index], "*")
    #     frame_list = sorted( glob.glob(video) )
    #     # print(frame_list)
    #     ground_truth_list = sorted(glob.glob(os.path.join(video_ground_truth_list[index],"*.bmp")))
    #     # print(ground_truth_list)
    #     img_size = cv2.imread(frame_list[0]).shape[0:2]

    #     # get video flow
    #     frame_flows = []
    #     for i in range( len(frame_list)-1 ):
    #         frame_flow = get_frame_flow(frame_list[i], frame_list[i+1], flownet, device, 512, 384)
    #         frame_flows.append(frame_flow)

    #     for i in range( len(frame_list)-4 ): #滑窗长度为5
    #         ground_truth_bbox = abnormal_bbox( cv2.imread(ground_truth_list[i], cv2.IMREAD_GRAYSCALE) )

    #         # frame roi
    #         roi = RoI(frame_list[i:i+5], "ped2" , yolo, device)
    #         normal_frame = np.array([[[]]])
    #         abnormal_frame = np.array([[[]]])
            
    #         for bbox in roi:
    #             object_imgs = np_load_frame_roi(frame_list[i], 64, 64, bbox)
    #             object_flows = flow_utils.flow2img ( roi_flow(frame_flows[i], bbox, 64, 64, img_size).cpu().numpy().transpose(1,2,0) )
    #             # print("object_img.shape: {}, object_flow.shape: {}".format(object_imgs.shape, object_flows.shape))
    #             for j in range(1,5):
    #                 object_img = np_load_frame_roi(frame_list[i+j], 64, 64, bbox)
    #                 object_imgs = np.concatenate([object_imgs, object_img], axis=1 )
    #             for j in range(1,4):
    #                 object_flow = flow_utils.flow2img (roi_flow(frame_flows[i], bbox, 64, 64, img_size).cpu().numpy().transpose(1,2,0) )
    #                 object_flows = np.concatenate([object_flows, object_flow], axis=1 )
    #             object_flows = np.concatenate([object_flows, np.zeros([64,64,3])], axis=1 )
    #             img = np.vstack((object_imgs, object_flows))

    #             # save
    #             if is_normal_object(ground_truth_bbox, bbox) == True:
    #                 try:
    #                     normal_frame = np.vstack((normal_frame, img))
    #                 except:
    #                     normal_frame = img
    #             else:
    #                 try:
    #                     abnormal_frame = np.vstack((abnormal_frame, img))
    #                 except:
    #                     abnormal_frame = img
                
    #         normal_path = os.path.join(save_dir, "normal", "ped2_"+ str(videos[index]) + "_" + str(i)+".jpg")
    #         abnormal_path = os.path.join(save_dir, "abnormal", "ped2_"+ str(videos[index]) + "_" + str(i)+".jpg")

    #         try:    
    #             cv2.imwrite( normal_path, normal_frame)
    #         except:
    #             pass 
    #         try:
    #             cv2.imwrite(abnormal_path, abnormal_frame)
    #         except:
    #             pass 
            
    #         print("save ped2 video: {} frame: {}".format(videos[index], i))
    #     break



    # video: avenue
    save_dir = "./show_object/avenue"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "normal"))
        os.mkdir(os.path.join(save_dir, "abnormal"))
    dataset = "/data0/lyx/VAD_datasets/avenue/testing/frames/"
    videos = sorted(os.listdir(dataset))
    videos_ground_truth = "/data0/lyx/VAD_datasets/avenue/testing_label_mask/"
    video_ground_truth_list = glob.glob( os.path.join(videos_ground_truth, "*.mat") )
    video_ground_truth_list = sorted( video_ground_truth_list)
    # print(video_ground_truth_list)
    
    for index in range(len(videos)):
        video = os.path.join(dataset, videos[index], "*")
        frame_list = sorted( glob.glob(video) )
        # print(frame_list)
        ground_truth_list = scipy.io.loadmat(video_ground_truth_list[index])['volLabel'][0]
        # print(ground_truth_list.shape)
        img_size = cv2.imread(frame_list[0]).shape[0:2]

        # get video flow
        frame_flows = []
        for i in range( len(frame_list)-1 ):
            frame_flow = get_frame_flow(frame_list[i], frame_list[i+1], flownet, device, 512, 384)
            cv2.imwrite( "./avenue_flow/test_01_frame_"+str(i) + ".jpg", flow_utils.flow2img(frame_flow.cpu().numpy().transpose(1,2,0)) )
            frame_flows.append(frame_flow)

        for i in range( len(frame_list)-4 ): #滑窗长度为5
            # ground truth bbox
            ground_truth_bbox = abnormal_bbox( ground_truth_list[i]*255 )

            # frame roi
            roi = RoI(frame_list[i:i+5], "avenue" , yolo, device)

            normal_frame = np.array([[[]]])
            abnormal_frame = np.array([[[]]])            
            
            for bbox in roi:
                object_imgs = np_load_frame_roi(frame_list[i], 64, 64, bbox)
                object_flows = flow_utils.flow2img ( roi_flow(frame_flows[i], bbox, 64, 64, img_size).cpu().numpy().transpose(1,2,0) )
                # print("object_img.shape: {}, object_flow.shape: {}".format(object_imgs.shape, object_flows.shape))
                for j in range(1,5):
                    object_img = np_load_frame_roi(frame_list[i+j], 64, 64, bbox)
                    object_imgs = np.concatenate([object_imgs, object_img], axis=1 )
                for j in range(1,4):
                    object_flow = flow_utils.flow2img (roi_flow(frame_flows[i], bbox, 64, 64, img_size).cpu().numpy().transpose(1,2,0) )
                    object_flows = np.concatenate([object_flows, object_flow], axis=1 )
                object_flows = np.concatenate([object_flows, np.zeros([64,64,3])], axis=1 )
                img = np.vstack((object_imgs, object_flows))

                # save
                if is_normal_object(ground_truth_bbox, bbox) == True:
                    try:
                        normal_frame = np.vstack((normal_frame, img))
                    except:
                        normal_frame = img
                else:
                    try:
                        abnormal_frame = np.vstack((abnormal_frame, img))
                    except:
                        abnormal_frame = img
                
            normal_path = os.path.join(save_dir, "normal", "avenue_"+ str(videos[index]) + "_" + str(i)+".jpg")
            abnormal_path = os.path.join(save_dir, "abnormal", "avenue_"+ str(videos[index]) + "_" + str(i)+".jpg")

            try:    
                cv2.imwrite( normal_path, normal_frame)
            except:
                pass 
            try:
                cv2.imwrite(abnormal_path, abnormal_frame)
            except:
                pass 
            
            print("save avenue video: {} frame: {}".format(videos[index], i))
        # break