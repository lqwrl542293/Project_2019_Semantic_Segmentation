import torch
from torch.autograd import Variable as V
from skimage.morphology import remove_small_objects
import torch.nn.functional as F
import cv2
import os
import numpy as np
from time import time
import argparse
from models.seg_hrnet import HighResolutionNet
from config import config
from config import update_config
from sklearn import metrics

def computeIoU(self, label_list=[], path1='', path2=''):
    union_sum = 0
    intersection_sum = 0
    IoU_list_class = []
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    for i in range(len(label_list)):
        i = label_list[i]
        temp1 = np.where(img1 == i, 1, 2)
        temp2 = np.where(img2 == i, 1, 3)
        intersection = np.sum(np.where(temp1 == temp2, 1, 0))
        intersection_sum += intersection  # 交集
        temp1 = np.where(img1 == i, 1, 0)
        temp2 = np.where(img2 == i, 1, 0)
        union = temp1 + temp2
        union[union > 1] = 1
        union = np.sum(union)
        union_sum += union  # 并集
        IoU_list_class.append(intersection / union)
        print('class ' + str(i) + ' IoU:', intersection / union)
    IoU = intersection_sum / union_sum
    return IoU, IoU_list_class

def testcutImg(self, source, target, patch_size=1024):  # source->src target->cut
    test_sets = os.listdir(source)
    original_shape = {}
    for i, image_name in enumerate(test_sets):  # randomly choose
        cut_image_folder = target + image_name[:-4] + '/'  # test/cut/.../
        if not os.path.exists(cut_image_folder):
            os.mkdir(cut_image_folder)
        img = cv2.imread(source + image_name)
        height, width, channel = img.shape  # attention!
        original_shape[image_name[:-4]] = img.shape  # should take control of the order
        image = np.zeros(shape=(height, width, channel))
        image[0:height, 0:width, :] = img
        start_width = 0
        end_width = start_width + patch_size
        start_height = 0
        end_height = start_height + patch_size

        count = 0
        while end_width <= width:
            while end_height <= height:
                patch = image[start_height:end_height, start_width:end_width, :]
                # print(count)
                cv2.imwrite(cut_image_folder + str(count) + '.png', img=patch.astype(np.uint8))
                # print(start_height, end_height, start_width, end_width)
                count += 1
                if (end_height == height):
                    break

                start_height += patch_size
                end_height = start_height + patch_size

                if end_height > image.shape[0]:
                    start_height = height - patch_size
                    end_height = height
            if(end_width ==  width):
                break

            start_width += patch_size
            end_width = end_width + patch_size
            if end_width > width:
                end_width = width
                start_width = width - patch_size
            # print(start_width, end_width)

            start_height = 0
            end_height = start_height + patch_size
        print('patches count:', count)
    print('original_shape_dict:', original_shape)
    return original_shape

def stitch_patches(self, source, target,  original_shape, patch_size=1024):
    '''
    :param source: 切割好的图片的文件夹
    :param target: 合成好的学出来的图片
    :param original_shape:
    :param patch_size:
    :return:
    '''
    # print(source)
    datasets = os.listdir(source)
    result = np.zeros(original_shape)
    for i, image in enumerate(datasets):
        # print(image)
        patch = cv2.imread(source + image)
        num = int(image[:-4])
        # print(num)
        (height, width, channel) = original_shape

        margin = int(height / patch_size) + 1  # height/patch_size
        # print(margin)
        start_width = (num // margin ) * patch_size  # //取整
        # print(start_width)
        end_width = start_width + patch_size
        # print(end_width)
        if end_width > width:
            end_width = width
            start_width = end_width - patch_size

        start_height = (num % margin) * patch_size
        # print(start_height)
        end_height = start_height + patch_size
        # print(end_height)
        if end_height > height:
            end_height = height
            start_height = end_height - patch_size

        # print(start_height,end_height, start_width,end_width)
        result[start_height:end_height, start_width:end_width, :] = patch
    result = result[0:original_shape[0], 0:original_shape[1], :]
    cv2.imwrite(target, img=result.astype(np.uint8))
    print("Stitch over!!!")


def visall(self, path, name):
    '''result -> result path'''
    # reslut = "/data/users/mzy/zyw/code/hrnet/dataset/Potsdam/train/label"
    # reslutlabel = os.listdir(result)
    # for each in reslutlabel:
        # if len(each) == 25:

    img = cv2.imread(path)
    print(img.shape)
    resultz = np.zeros(shape=img.shape)
    channel0 = np.where(img[:, :, 0] == 0, 255, 0)  # blue
    resultz[:, :, 0] = channel0
    resultz[:, :, 1] = channel0
    resultz[:, :, 2] = channel0
    channel1 = np.where(img[:, :, 0] == 1, 255, 0)  # blue
    resultz[:, :, 2] = resultz[:, :, 2] + channel1

    channel2 = np.where(img[:, :, 0] == 2, 255, 0)  # blue
    resultz[:, :, 1] = resultz[:, :, 1] + channel2
    resultz[:, :, 2] = resultz[:, :, 2] + channel2

    channel3 = np.where(img[:, :, 0] == 3, 255, 0)  # blue
    resultz[:, :, 1] = resultz[:, :, 1] + channel3

    channel4 = np.where(img[:, :, 0] == 4, 255, 0)  # blue
    resultz[:, :, 0] = resultz[:, :, 0] + channel4
    resultz[:, :, 1] = resultz[:, :, 1] + channel4

    channel5 = np.where(img[:, :, 0] == 5, 255, 0)  # blue
    resultz[:, :, 0] = resultz[:, :, 0] + channel5

    print(cv2.imwrite(source + 'multilabel/' + name + '.png', resultz))

    print(name,"wirten down")
