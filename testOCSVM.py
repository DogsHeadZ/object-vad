import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
import cv2
from tqdm import tqdm
from vad_dataloader import VadDataset
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn


batch_size = 16
dataset = 'ped2'
datadir = "../AllDatasets/"+dataset+"/testing/frames"

train_data = VadDataset(datadir, "./bboxes/"+dataset+"/test",
                        transform=transforms.Compose([transforms.ToTensor()]),
                        resize_height=224, resize_width=224)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

unloader = transforms.ToPILImage()

feature_extractor = resnet50(pretrained=True).eval().cuda()
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1]) # 去掉全连接层做为特征提取器

X = next(iter(train_dataloader))
print(X.shape)
feats = []
for j, imgs in enumerate(tqdm(train_dataloader, desc='train', leave=False)):
    imgs = imgs.cuda()
    imgs = imgs[:, 6:9]
    feat = feature_extractor(imgs).squeeze()
    feats.append(feat.cpu().detach().numpy())
    if j == 210:
        break

feats = np.concatenate(feats, axis=0)
print(feats.shape)

ts = TSNE(n_components=2)
# 训练模型
y = ts.fit_transform(feats)
print(ts.embedding_.shape)

plt.scatter(y[:, 0], y[:, 1])
# ax1.set_title('t-SNE Curve', fontsize=14)
# 显示图像
plt.savefig('tsne.jpg')