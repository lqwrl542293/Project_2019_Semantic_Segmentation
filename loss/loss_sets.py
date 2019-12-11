#encoding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F


#单分类loss 输出单通道
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        return a + b

#多分类loss

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, target, preds):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss


class CrossEntropy(nn.Module):

    def __init__(self, ignore_label=-1):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, target, score):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')
        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.9, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.thresh = thres
        # self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, target, score,  **kwargs):

        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        # print(h,w)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1) #HRNet最后没过softmax，在这里输出概率
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        # min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = self.thresh
        # threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.063784, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target,input):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W 16,,1024,1024
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) #16,1024,1024

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target) #input[i][index[i][j]]如果网络输出是二通道，需要根据GT把他们压缩到N*1，单通道直接压缩
        logpt = logpt.view(-1)
        pt = V(logpt.data.exp())

        if self.alpha is not None:
            # if self.alpha.type()!=input.data.type():
            #     self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * V(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

