#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torchvision.transforms as transforms

from lite_flownet import Network, batch_estimate, estimate

import cv2
import numpy as np

lite_flow_model_path = 'network-sintel.pytorch'

flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval()
image1 = 'images/005.jpg'
image2 = 'images/006.jpg'
resize_width = 256
resize_height = 256
transform = transforms.Compose([
        transforms.ToTensor(),
    ])

image_r1 = cv2.imread(image1)
image_r1 = transform(image_r1).unsqueeze(dim=0).cuda()
image_r2 = cv2.imread(image2)
image_r2 = transform(image_r2).unsqueeze(dim=0).cuda()

image_resize1 = cv2.imread(image1)
image_resize1 = cv2.resize(image_resize1, (resize_width, resize_height))
image_resize1 = transform(image_resize1).unsqueeze(dim=0).cuda()
image_resize2 = cv2.imread(image2)
image_resize2 = cv2.resize(image_resize2, (resize_width, resize_height))
image_resize2 = transform(image_resize2).unsqueeze(dim=0).cuda()

# image_decoded1 = cv2.imread(image1)
# image_resized1 = cv2.resize(image_decoded1, (resize_width, resize_height))
# image_resized1 = image_resized1.astype(dtype=np.float32)
# image_resized1 = (image_resized1 )/255.0

flow_r_tensor = torch.cat([image_r1, image_r2], 1)
flow_r_out = batch_estimate(flow_r_tensor, flow_network)[0]
objectOutput = open('out/flow_r.flo', 'wb')
np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
np.array([flow_r_out.size(2), flow_r_out.size(1)], np.int32).tofile(objectOutput)
np.array(flow_r_out.detach().cpu().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
objectOutput.close()

flow_resize_tensor = torch.cat([image_resize1, image_resize2], 1)
flow_resize_out = batch_estimate(flow_resize_tensor, flow_network)[0]
objectOutput = open('out/flow_resize.flo', 'wb')
np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
np.array([flow_resize_out.size(2), flow_resize_out.size(1)], np.int32).tofile(objectOutput)
np.array(flow_resize_out.detach().cpu().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
objectOutput.close()


tensorFirst = torch.FloatTensor(
    numpy.array(PIL.Image.open('images/first.png'))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0))

tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open('images/second.png'))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

tensorOutput = estimate(tensorFirst, tensorSecond)
objectOutput = open('out/flow_ori.flo', 'wb')

numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
numpy.array(tensorOutput.detach().numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

objectOutput.close()