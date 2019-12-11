import argparse
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from models.seg_hrnet1 import HighResolutionNet
from config import config
from config import update_config
from torchsummary import summary

#更新参数代码
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    update_config(config, args)

    return args


args = parse_args()



model = HighResolutionNet(config)
# model_dict = model.state_dict()
# summary(model,(2, 3 , 224, 224 ))
pretrained = '/data/users/hwt/zyw/code/tiamzhibei/hrnet_w18_small_v2_cityscapes_cls19_1024x2048_trainset.pth'
pretrained_dict = torch.load(pretrained)
# print(pretrained_dict.shape)
new_state_dict = collections.OrderedDict()
for k, v in pretrained_dict.items():
    # name = 'module.'+k
    name = k.replace('model.', '')  # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

for k, v in new_state_dict.items():
    print(k,v.shape)
# pretrained_dict = torch.load(pretrained)
# # print(pretrained_dict)
# # model_dict = model.state_dict()
# model.load_state_dict(pretrained_dict)

#     if k == 'module.conv1.weight':
#         print(v)
#         # print(j.shape)
#         # print()
#         # print(.shape)
#         # print(j[:,1:2,:,:].shape)
#         # print(j[:, 2:3, :, :].shape)
#         a = v[:, 0:1, :, :] + v[:, 1:2, :, :] + v[:, 2:3, :, :]
#         a = a/3
#         print(a.shape)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
#                                bias=False)
# model.conv1.weight = torch.nn.Parameter(a[:,:,:,:])
# torch.save(model.state_dict(), '/data/users/hwt/zyw/code/tiamzhibei/weights/HRNet_small_multiclass_trick_single.th')
# #

# model_dict = model.state_dict()
# print()
# print()
# for k, v in pretrained_dict.items():
#     print(k,v.shape)



# pretrained_dict = {k[:] : v }
# model.cuda()
# print(pretrained_dict)
# para = torch.load("G:\tianzhibei\weights\HRNet_small_multiclass_trick.th")
# if para.key == 'conv1.weight':
#     print(para[para.key].shape)

# print(model.state_dict().keys())
# for i, j in model.named_parameters():
#
#     if i =='conv1.weight':
#         # print(j)
#         # print(j.shape)
#         # print()
#         # print(.shape)
#         # print(j[:,1:2,:,:].shape)
#         # print(j[:, 2:3, :, :].shape)
#         a = j[:, 0:1, :, :] + j[:, 1:2, :, :] + j[:, 2:3, :, :]
#         a = a/3
#         print(a.shape)

        # print(j[:,,:,:])