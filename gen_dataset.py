#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 1024 #生成图片的宽和高

img_h = 1024

image_sets = [['TZB_4','TZB_5','TZB_6','TZB_7','TZB_8','TZB_10','TZB_11','TZB_12'], ['TZB_1','TZB_2','TZB_3']]#需要生成的图片


red = np.array([255,0,0])
green = np.array([0,255,0])
blue = np.array([0,0,255])
yellow = np.array([255,255,0])



def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask, mode,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if mode == 'color':
            height, width, channel = image.shape
        if mode == 'gray':
            height, width = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask
    
    
def data_augment(xb,yb, mode): #xb:source;yb:label
    if np.random.random() < 0.25: #随机旋转
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)

    if mode != 'gray':
    # hsv空间颜色变化
        xb = randomHueSaturationValue(xb,hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    xb, yb = randomShiftScaleRotate(xb, yb, mode,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0)) #留作实验，测试pipeline

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() <0.25: #沿着x轴旋转
        xb = cv2.flip(xb,0)
        yb = cv2.flip(yb,0)
        
    if (np.random.random() < 0.1) & (mode != 'gray'):
        xb = random_gamma_transform(xb,1.0)

        
    return xb,yb


def creat_dataset(image_num = 16000, mode = 'color'):
    if mode == 'color':
        idx = 0
    else:
        idx = 1
    ROOT_DIR = '/data/public/datasets/tianzhi_kemu2/tianzhi/' #如果文件结构变了需要更改
    print('creating dataset...')
    image_each = image_num / len(image_sets[idx])
    g_count = 0
    for i in tqdm(range(len(image_sets[idx]))):
        count = 0
        if mode == 'gray':
            #print(f"{index}, {i}")
            src_img = cv2.imread(ROOT_DIR + 'src/' + image_sets[idx][i] + '.tif', cv2.IMREAD_GRAYSCALE)#如果文件结构变了需要更改
            X_height, X_width= src_img.shape
        else:
            src_img = cv2.imread(ROOT_DIR+'src/' + image_sets[idx][i]+'.tif')  # 3 channels #如果文件结构变了需要更改
            X_height, X_width, _ = src_img.shape
        label_img = cv2.imread(ROOT_DIR+'label_onechannel/' + image_sets[idx][i]+'.png',cv2.IMREAD_GRAYSCALE)  # single channel
        # 如果文件结构变了需要更改
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1) #随机取得起始点
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            #if mode == 'augment':
            src_roi,label_roi = data_augment(src_roi,label_roi,mode)

            colormap = np.zeros((img_h, img_w, 3))
            index = np.where(label_roi == 1)
            colormap[index] = red
            index = np.where(label_roi == 2)
            colormap[index] = green
            index = np.where(label_roi == 3)
            colormap[index] = blue
            index = np.where(label_roi == 4)
            colormap[index] = yellow
            colormap = colormap[:,:,::-1]
            
            cv2.imwrite(ROOT_DIR+'aug_train/'+mode+('/visualise/%d.png' % g_count),colormap)#如果文件结构变了需要更改
            cv2.imwrite(ROOT_DIR+'aug_train/'+mode+('/src/%d.png' % g_count),src_roi)#如果文件结构变了需要更改
            cv2.imwrite(ROOT_DIR+'aug_train/'+mode+('/label/%d.png' % g_count),label_roi)#如果文件结构变了需要更改
            count += 1 
            g_count += 1


            
    

if __name__=='__main__':  
    creat_dataset(mode='color')
