import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import datetime


class KernelPredeictionModule(nn.Module):

    def __init__(self, input_channel, channel_cm=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        super(KernelPredeictionModule,self).__init__()
        self.input_channel = input_channel
        self.channel_cm = channel_cm
        self.kernel_up = kernel_up
        self.kernel_encoder = kernel_encoder
        self.enlarge_rate = enlarge_rate
        self.channel_compressor = nn.Sequential(
            OrderedDict([
                ("compressor_conv" , nn.Conv2d(self.input_channel, self.channel_cm,1,1,0,bias=False)),
                ("compressor_bn"   , nn.BatchNorm2d(self.channel_cm)),
                # ("compressor_relu" , nn.ReLU(inplace=True))
            ])
        )
        self.context_encoder = nn.Sequential(
            OrderedDict([
                ("encoder_conv"    , nn.Conv2d(self.channel_cm,
                                          self.enlarge_rate*self.enlarge_rate*self.kernel_up*self.kernel_up,# rate^2*kup^2
                                          self.kernel_encoder,padding=int((self.kernel_encoder-1)/2),bias=False)),
                # ("encoder_bn"      , nn.BatchNorm2d(self.enlarge_rate*self.enlarge_rate*self.kernel_up*self.kernel_up)),
                # ("encoder_relu"    , nn.ReLU(inplace=True))
            ])
        )
        self.kernel_normalizer = nn.Softmax(dim=1)
    def forward(self, x):
        # start_time = datetime.datetime.now()
        x = self.channel_compressor(x)
        x = self.context_encoder(x)
        x = F.pixel_shuffle(x,self.enlarge_rate)
        x = self.kernel_normalizer(x)
        # print("KP cost:{}".format(datetime.datetime.now() - start_time))
        return x

class Carafe(nn.Module):
    def __init__(self, input_channel, channel_cm=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        """
        The Carafe upsample model(unoffical)
        :param input_channel: The channel of input
        :param channel_cm:    The channel of Cm, paper give this parameter 64
        :param kernel_up:     The kernel up, paper give this parameter 5
        :param kernel_encoder:The kernel encoder, paper suggest it kernel_up-2, so 3 here
        :param enlarge_rate:  The enlarge rate , your rate for upsample (2x usually)
        """
        super(Carafe, self).__init__()
        self.kernel_up = kernel_up
        self.enlarge_rate = enlarge_rate
        self.KPModule = KernelPredeictionModule(input_channel,channel_cm,kernel_up,kernel_encoder,enlarge_rate)

    def forward(self, x):

        # KernelPredeictionModule : cost 0.7175s
        kpresult = self.KPModule(x) # (b,kup*kup,e_w,e_h)


        ############Context-aware Reassembly Module########################
        ######## Step1 formal_pic deal : cost 0.1164s
        x_mat = self.generate_kup_mat(x) # (b,c,kup*kup,w,h)

        ######## Step2 kernel mul : cost 0.0009s
        output = x_mat * (kpresult.unsqueeze(1))

        ######## Step3 sum the kup dim : cost 0.0002s
        output = torch.sum(output, dim=2)
        return output

    def generate_kup_mat(self,x):
        """
        generate the mat matrix, make a new dim kup for mul
        :param x:(batch,channel,w,h)
        :return: (batch,channel,kup*kup,enlarged_w,enlarged_h)
        """
        batch, channel, w ,h = x.shape
        # stride to sample
        r = int(self.kernel_up / 2)


        # get the dim kup**2 with unfold with the stride windows
        x_mat = torch.nn.functional.unfold(x, kernel_size=self.kernel_up, padding=r, stride=1)
        # make the result to (b,c,kup**2,w,h)
        x_mat = x_mat.view((batch, channel, self.kernel_up**2, w, h))
        # nearest inter the number for i map the region [i:i/enlarge,j:j/enlarge]
        x_mat = torch.nn.functional.interpolate(x_mat,
                                                scale_factor=(1, self.enlarge_rate, self.enlarge_rate),
                                                mode='nearest')
        return x_mat

# speed test
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = Carafe(input_channel=320,channel_cm=64).cuda()
    x = torch.rand((16,320,32,24)).cuda()
    # from model_summary import summary
    # summary(model,x)

    model = model.cuda()
    x = x.cuda()
    print(x[0].size(0))
    model(x)
    start_time = datetime.datetime.now()
    out = model(x)
    print("time cost:{}".format(datetime.datetime.now()-start_time))
    print(out.shape)

    time2 = datetime.datetime.now()
    out2 = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
    print("time cost_upsample:{}".format(datetime.datetime.now()-time2))
    print(out2.shape)

    time3 = datetime.datetime.now()
    deconv_layer = nn.ConvTranspose2d(320, 320, 3, stride=2, padding=1, output_padding=1).cuda()
    out3 = deconv_layer(x)
    print("time cost_upsample:{}".format(datetime.datetime.now() - time3))
    print(out3.shape)
