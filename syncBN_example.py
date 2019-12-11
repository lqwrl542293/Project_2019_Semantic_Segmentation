import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse

# the default is using two gpus 0 and 1

batch_size = 256
#os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),])
)

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.BN1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=4,stride=1)
        self.BN3 = nn.BatchNorm2d(120)

        self.fc1 = nn.Linear(in_features=120,out_features=84,bias=True)
        self.fc2 = nn.Linear(in_features=84,out_features=10,bias=True)

    def forward(self,x):

        x = self.conv1(x)
        x = self.BN1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.BN2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.BN3(x)
        x = F.relu(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--ngpu', type=int, default=4)
args = parser.parse_args()

# step 1
world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

# step 2
sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=args.ngpu,
    rank=args.local_rank,
)
data_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,num_workers=0,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)
##pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。

# step 3
torch.cuda.set_device(args.local_rank)
net = Lenet()
net.cuda()
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = torch.nn.parallel.DistributedDataParallel(
    net,
    #device_ids=[args.local_rank],
    output_device=args.local_rank,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#train
for i in range(10):
    for it, (data, target) in enumerate(data_loader):
        data,target = data.cuda(),target.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()