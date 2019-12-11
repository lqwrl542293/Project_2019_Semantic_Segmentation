#coding=utf-8
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, config, args,lr=2e-4,evalmode = False):
        if config.USESYNCBN is True:
            print('parallel training model setup')
            # gpu_map = [0, 5, 6, 7]
            gpu_map = [0, 1, 2, 3]
            torch.cuda.set_device(gpu_map[args.local_rank])  # set_device to multi_gpu
            self.net = net.cuda()
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            # self.net = torch.nn.DataParallel(self.net)  #device_ids=range(torch.cuda.device_count())) #指定master GPU
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[gpu_map[args.local_rank]],
                output_device=gpu_map[args.local_rank],
            )
        else:
            print('normal training')
            self.net = net.cuda()
            self.net = torch.nn.DataParallel(self.net)
        if evalmode:
            self.net.eval()
        else:
            self.net.train()
        if config.TRAIN.OPTIMIZER == 'sgd':

            self.optimizer = torch.optim.SGD(
                [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': config.TRAIN.LR}],
                lr=config.TRAIN.LR, momentum=config.TRAIN.MOMENTUM, weight_decay=config.TRAIN.WD)
        else:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode: #评估模式，冻结BN的统计量
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img): #测试单张图片
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward()
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        '''
        In dlinknet, each conv-layer is followed by a batchnorm(conv-bn-relu), and, in fact, there's no difference between [-1.6,1.6] and [-1,1](or even [0,255]) in this case.
        However, Unet(weight initialized by pytorch default initializer) without batchnorm do have better result using [-1.6,1.6] normalization. In this case, [-1.6,1.6](compare with [-1,1]) 
        is equal to enlarge the initial learning rate and std of the initializer.
        In dlinknet, channel-wise normalization equals adding two parameters weighting the RGB. It may have better result(I'm not sure), but I don't think this is so crucial~ 2333
        '''
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self):
        #self.img = V(self.img.cuda(), volatile=volatile)
        self.img = self.img.cuda()
        if self.mask is not None:
            #self.mask = V(self.mask.cuda(), volatile=volatile)

            self.mask = self.mask.cuda()
        
    def optimize(self, config):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net(self.img)
        if config.TRAIN.LOSS == 'CrossEntropy' or config.TRAIN.LOSS == 'FocalLoss' or config.TRAIN.LOSS == 'Ohem':
            self.mask.squeeze_()
            self.mask = self.mask.to(torch.long)
        #print(self.mask.dtype)


        # self.mask = self.mask.cpu()
        # pred = pred.cpu()

        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        pretrained_dict = torch.load(path)

        new_state_dict = collections.OrderedDict()
        for k, v in pretrained_dict.items():
            # name = 'module.'+k
            name = k.replace('module', '')  # remove `module.`
            name = 'module' + name
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
    
    def update_lr(self, new_lr, mylog, factor=False): # factor = true 之后，new_lr成为缩小几倍的表示
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr), file = mylog)
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

