import os
from PIL import Image
import numpy as np
import cv2

# def file_paths(root_dir):
#     L = []
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == '.png':
#                 L.append(os.path.join(root,file))
#     return L
#
#
# def file_names(root_dir):
#     L = []
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             L.append(os.path.split(file)[1][0:-4])
#     return L
#
# label_root = '/Users/muzhengyang/Documents/tianzhi/label'
# src_root = '/Users/muzhengyang/Documents/tianzhi/src'
# files_label = file_names(label_root)
#
# for name in files_label:
#     label_path = label_root+'/'+name+'.png'
#     img_path = src_root+'/'+name[:-4]+'.tif'
#
#     bottom = cv2.imread(img_path)
#     top = cv2.imread(label_path)
#
#     # 权重越大，透明度越低
#     overlapping = cv2.addWeighted(bottom, 0.5, top, 0.5, 0)
#     # 保存叠加后的图片
#     cv2.imwrite(name+'overlap(8:2).jpg', overlapping)
#     print('finish::::'+name)


#files = file_paths('/Users/muzhengyang/Documents/tianzhi/label')
file = '/Users/muzhengyang/Desktop/天智杯/BDCI2017-jiage/CCF-training/2_class_8bits.png'
#print(files)

red = np.array([255,0,0])
green = np.array([0,255,0])
blue = np.array([0,0,255])
yellow = np.array([255,255,0])

#for file in files:
img = Image.open(file)
tmp = np.array(img)
a = np.unique(tmp)
h_size, w_size = tmp.shape
colormap = np.zeros((h_size,w_size,3))
index = np.where(tmp==1)
colormap[index] = red
index = np.where(tmp==2)
colormap[index] = green
index = np.where(tmp==3)
colormap[index] = blue
index = np.where(tmp==4)
colormap[index] = yellow
savefile = Image.fromarray(np.uint8(colormap)).convert('RGB')
path = '/Users/muzhengyang/Desktop/天智杯/BDCI2017-jiage/CCF-training/label2'+'_rgb.png'
savefile.save(path)
#print(os.path.splitext(file)[0]+'_rgb.png')



#
#
#
#
