import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np 
from PIL import Image
import os
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




def plot_features(features):
    # 对特征做TSNE降维，可视化
    features_embedded = TSNE(n_components=2).fit_transform(features)
    print(features_embedded.shape)
    plt.scatter(features_embedded[:,0], features_embedded[:,1])
    plt.savefig('tsne2.jpg')

def get_clips(frame_dir, frames, bbox):
    """load a clip to be fed to C3D
    a pytorch batch: (n, ch, fr, h, w) """
    clips = []
    for frame in frames:
        clip = Image.open(os.path.join(frame_dir, frame))
        clip = clip.crop((bbox[0],bbox[2],bbox[1],bbox[3]))
        clip =  clip.resize((112,112))
        clips.append(np.array(clip).astype(np.float32))

    clips = np.array(clips).astype(np.float32)
    clips = clips.transpose(3, 0, 1, 2)
    clips = torch.tensor(clips)
    clips = torch.unsqueeze(clips, dim=0)
    
    return  clips

def resnet_features(frame_dir, bboxes, device):
    #### extract object features via resnet50 
    model = models.resnet50(pretrained = True)
    extractor = nn.Sequential(*list(model.children())[:-1]) # 去掉卷积层做为特征提取器
    extractor.to(device)
    # print(extractor)

    extractor.eval()
    features = []
    frame_list = sorted(os.listdir(frame_dir))
    for i in range(2,len(frame_list)-2):
        frame = Image.open(os.path.join(frame_dir, frame_list[i]))
        # frame = cv2.imread(os.path.join(frame_dir, frame_list[i]))
        for bbox in bboxes[i-2]:
            img = frame.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
            (xmin, ymin, xmax, ymax) = bbox

            # img = frame[ymin:ymax, xmin:xmax]
            # img = cv2.resize(img, (224, 224))
            img = transform(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)

            feature = extractor(img).flatten()
            features.append(feature.cpu().detach().numpy())
    return np.array(features)  


def C3D_features(frame_dir, bboxes, device):
    #### extract object features via pretrained C3D, 16 frames
    from C3D import C3D
    model = C3D(num_classes=101, pretrained=True)
    # print(model)
    model.to(device)
    ########## imshow clips #############
    # index = 1
    # for bbox in bboxes[10]:
    #     for i in range(10,26):    
    #         frame = Image.open(os.path.join(frame_dir, frame_list[i]))
    #         img = frame.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
    #         plt.subplot(len(bboxes[10]), 16, index)
    #         plt.axis('off')
    #         plt.imshow(img)
    #         index += 1
    # plt.show()

    features = []
    frame_list = sorted(os.listdir(frame_dir))
    for i in range(len(frame_list)-15):
        for bbox in bboxes[i]:
            clips = get_clips(frame_dir, frame_list[i:i+16], bbox)
            clips = clips.to(device)
            feature = model.extract(clips).flatten()
            features.append(feature.cpu().detach().numpy())
    return np.array(features)


if __name__ == "__main__":
    # test 
    frame_dir = "../AllDatasets/ped2/testing/frames/01"
    bboxes = np.load("./bboxes/ped2/test/01.npy", allow_pickle=True)
    device = "cuda"
    
    # resnet 
    features = resnet_features(frame_dir, bboxes, device)
    print("plot resnet features")
    plot_features(features)

    #c3d
    # features = C3D_features(frame_dir, bboxes, device)
    # print("plot C3D features")
    # plot_features(features)



