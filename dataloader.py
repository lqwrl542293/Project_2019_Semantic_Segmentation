#coding=utf-8
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os


def default_loader(id, root, trainclass=1):
    # print(os.path.join(root,'{}.png').format(id))
    img = cv2.imread((root + '/src/{}.png').format(id))

    # print(os.path.join(root+'/label/{}.png').format(id))
    mask = cv2.imread((root + '/label/{}.png').format(id), cv2.IMREAD_GRAYSCALE)

    # 扩充mask维度为三维数组
    mask = np.expand_dims(mask, axis=2)
    #print('img data shape---- {}'.format(img.shape))
    # 正则化项 transpose 使得维度与pytorch约定的对齐


    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    if trainclass!=-1 :
        mask[mask != trainclass] = 0
        mask[mask == trainclass] = 1
    # mask = abs(mask-1)
    return img, mask





class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root, trainclass= -1): #如果train class设置为-1，则代表多分类
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.segclass = trainclass

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root,trainclass=self.segclass)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)

