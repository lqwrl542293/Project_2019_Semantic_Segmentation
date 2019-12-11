from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os,sys,random,time
import argparse

from loss_sets import FocalLoss


# start_time = time.time()
# maxe = 0
# for i in range(1000):
#     x = torch.rand(12800,2)*random.randint(1,10)
#     x = Variable(x.cuda())
#     l = torch.rand(12800).ge(0.1).long()
#     l = Variable(l.cuda())
#
#     output0 = FocalLoss(gamma=0)(x,l)
#     output1 = nn.CrossEntropyLoss()(x,l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a-b)>maxe: maxe = abs(a-b)
# print('time:',time.time()-start_time,'max_error:',maxe)


start_time = time.time()
maxe = 0
#
x = torch.rand(16,2,1024,1024)*random.randint(1,10)
x = Variable(x)
l = torch.rand(16,1024,1024)*2    # 1000 is classes_num
l = l.long()
l = Variable(l)

output0 = FocalLoss(gamma=2,alpha=0.25)(x,l)
output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
a = output0.item()
b = output1.item()
if abs(a-b)>maxe: maxe = abs(a-b)
print('time:',time.time()-start_time,'max_error:',maxe)
