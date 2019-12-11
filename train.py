#coding=utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import torch.backends.cudnn as cudnn
import models

from models.seg_hrnet import HighResolutionNet
from models.ccnet import ResNet
import argparse
from time import time
import sys
from framework import MyFrame
from loss.loss_sets import dice_bce_loss, CrossEntropy, FocalLoss, OhemCrossEntropy, CriterionDSN
from dataloader import ImageFolder
from tensorboardX import SummaryWriter
from config import config
from config import update_config
from test import TTAFrame
# from loss.criterion import CriterionDSN, CriterionOhemDSN


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

print("-----start------")

#参数配置
# cudnn related setting
cudnn.benchmark = config.CUDNN.BENCHMARK
cudnn.deterministic = config.CUDNN.DETERMINISTIC
cudnn.enabled = config.CUDNN.ENABLED

SHAPE = config.TRAIN.IMAGE_SIZE # size of input image
ROOT = config.ROOT #训练数据的存储位置
#imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
BATCHSIZE_PER_CARD = config.TRAIN.BATCHSIZE_PER_CARD
CARD_NUM = config.TRAIN.CARD_NUM
lr = config.TRAIN.LR
total_epoch = config.TRAIN.TOTALEPOCH

if config.USESYNCBN is True:
    ##syncBN分布式参数
    print("use syncBN")
    world_size = CARD_NUM  #GPU个数
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=args.local_rank,
    )


#常数
no_optim = 0
train_epoch_best_loss = 100.

if os.path.exists('log/'):
    mylog = open('log/' + config.EXPNAME+'_' +str(args.local_rank) + '.log', 'a')
else:
    os.mkdir('log')
    mylog = open('log/' + config.EXPNAME + '_' +str(args.local_rank) +'.log', 'a')


imagelist = os.listdir(ROOT+"/src")
trainlist = list(map(lambda x: x[:-4], imagelist)) #去后缀
# print(trainlist)
# 如果使用HRNet
if config.MODEL.NAME =='seg_hrnet':
    print("preparing HRNet")
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)
elif config.MODEL.NAME == 'ccnet':
    # if config.MODEL.PRETRAINED
    criterion = CriterionDSN()  # CriterionCrossEntropy()
    model = eval('models.' + config.MODEL.NAME + '.Seg_Model')(
        num_classes=config.DATASET.NUM_CLASSES, criterion=criterion,
        pretrained_model=config.MODEL.PRETRAINED
    )

else:
    print('prepareing {}'.format(config.MODEL.NAME))
    model = eval(config.MODEL.NAME+'()')

#eval('solver = MyFrame(model, '+'config.TRAIN.LOSS'+', lr)')
# #配置framework,多分类的话改第二个参数，从loss_set中去选
if config.MODEL.NAME == 'ccnet':
    solver = MyFrame(model, CriterionDSN, config, args, lr)
else:
    if config.TRAIN.LOSS == 'dice_bce_loss':
        solver = MyFrame(model,dice_bce_loss, config, args,lr)
    elif config.TRAIN.LOSS == 'CrossEntropy':
        solver = MyFrame(model, CrossEntropy, config, args, lr)
    elif config.TRAIN.LOSS == 'FocalLoss':
        solver = MyFrame(model, FocalLoss, config, args, lr)
    elif config.TRAIN.LOSS == 'Ohem':
        loss = OhemCrossEntropy(ignore_label=-1, thres=config.LOSS.OHEMTHRES)
        solver = MyFrame(model, loss, config, args, lr)
    else:
        raise RuntimeError

if config.TRAIN.RESUME is True:
    print('resume from {}'.format(config.TRAIN.RESUME_START))
    print('resume from {}'.format(config.TRAIN.RESUME_START), file=mylog)
    solver.load(config.MODEL.PRETRAINED)

dataset = ImageFolder(trainlist, ROOT, trainclass=config.TRAIN.TRAINCLASS)#如果train class设置为-1，则代表多分类

# solver = MyFrame(xxx, CrossEntropy, lr) 多分类
if config.USESYNCBN is True:
    print('syncBN dataloader')
    batchsize = BATCHSIZE_PER_CARD #计算batch大小
    # syncBN分布式
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=args.local_rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # shuffle=config.TRAIN.SHUFFLE,
        batch_size=batchsize,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=False,
        sampler=sampler,
        drop_last=True)
else:
    a = time()
    batchsize = CARD_NUM * BATCHSIZE_PER_CARD
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=config.TRAIN.NUM_WORKERS,
        drop_last=True,
        pin_memory = True) # 加了这个测试一下会不会变快
    print("dataloadertime", time() - a)




tic = time()

writer = SummaryWriter('runs/'+ config.EXPNAME)

#TODO  只存储rank0的模型和输出即可


for epoch in range(1, total_epoch + 1):
    print('---epoch start-----')
    #data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader:

        solver.set_input(img, mask)
        train_loss = solver.optimize(config)
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader)
    if args.local_rank == 0:
        print('********',file=mylog)
        print('epoch:'+ str(epoch+config.TRAIN.RESUME_START) + '    time:'+ str(time()-tic), file=mylog)
        print('train_loss: {}'.format(train_epoch_loss), file=mylog)
        writer.add_scalar('scalar/train',train_epoch_loss,epoch+config.TRAIN.RESUME_START)
        print('********')
        print('epoch:'+str(epoch+config.TRAIN.RESUME_START)+'    time:'+ str(time()-tic))
        print('train_loss: {}'.format(train_epoch_loss))
        split = False
        if epoch%10 == 0:

            BATCHSIZE_PER_CARD = config.TEST.BATCH_SIZE_PER_GPU
            label_list = config.TEST.LABEL_LIST
            source = config.TEST.ROOT
            if config.MODEL.NAME == 'seg_hrnet':
                test = TTAFrame(HighResolutionNet, 'HRNet', label_list, config=config)
            else:
                test = TTAFrame(ResNet, 'ccnet', label_list, config=config)
            test.load(config.TEST.WEIGTH)
            kappa,  oa = test.test(source=source, label_list=label_list, split = split)
            split = True
            # print(text)
            print("kappa: " + str(kappa) + '\r\n')
            print("oa: " + str(oa) + '\r\n')

            # print(text, file=mylog)
            print("kappa: " + str(kappa) + '\r\n', file=mylog)
            print("oa: " + str(oa) + '\r\n', file=mylog)
            writer.add_scalar("Test/Accu(oa)", oa, epoch)
            writer.add_scalar("Test/Accu(kappa)", kappa, epoch)
            # print('SHAPE: {}'.format(SHAPE))

        #模型保存策略，停止条件和，调整lr_rate的时间
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+config.EXPNAME+'.th')#+'__'+str(args.local_rank)
    if no_optim > 10:
        print('early stop at %d epoch' % (epoch+config.TRAIN.RESUME_START), file=mylog)
        print('early stop at %d epoch' % (epoch+config.TRAIN.RESUME_START))
        break
    if no_optim > 4:
        if solver.old_lr < 5e-7:
            break
        #solver.load('weights/'+config.EXPNAME+'.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

writer.close()
print ('Finish!',file=mylog)
print('Finish!')
mylog.close()
