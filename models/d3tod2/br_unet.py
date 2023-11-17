import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DoubleConvCell(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convcell = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(mid_channels),
            nn.Dropout2d(p=0.25),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels),
            nn.Dropout2d(p=0.25),
            nn.ReLU(inplace=True))

    def forward(self, x):           
        return self.double_convcell(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.double_conv = DoubleConvCell(in_channels, out_channels)

    def forward(self, x):
        for i in range(x.shape[1]):
            f = x[:,i,:,:,:]          
            # 'b, 1, h, w' [64,1,64,64]
            conv = self.double_conv(f)
            conv = conv.unsqueeze(1)
            if i == 0:
                conv_result = conv               
            else:
                conv_result = torch.concat([conv_result, conv], 1)
                
        return conv_result

class BiConvGRUCell(nn.Module):
    """
        ICLR2016: Delving Deeper into Convolutional Networks for Learning Video Representations
        url: https://arxiv.org/abs/1511.06432
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, cuda_flag=True):
        super(BiConvGRUCell, self).__init__()
        self.input_channels  = input_channels
        self.cuda_flag  = cuda_flag
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        padding = self.kernel_size // 2
        self.reset_gate  = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.output_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        # init
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        if hidden is None:
           size_h    = [x.data.size()[0], self.hidden_channels] + list(x.data.size()[2:])
           if self.cuda_flag:
              hidden = torch.zeros(size_h).cuda()
           else:
              hidden = torch.zeros(size_h)

        inputs       = torch.cat((x, hidden), dim=1)
        reset_gate   = torch.sigmoid(self.reset_gate(inputs))
        update_gate  = torch.sigmoid(self.update_gate(inputs))

        reset_hidden = reset_gate * hidden        
        reset_inputs = torch.tanh(self.output_gate(torch.cat((x, reset_hidden), dim=1)))        
        new_hidden   = update_gate*reset_inputs + (1 - update_gate)*hidden

        return new_hidden

class BiConvGRU(nn.Module):
    def __init__(self, input_channels) :
        super().__init__()
        self.convgru_forward = BiConvGRUCell(input_channels, input_channels, 3)
        self.convgru_backward = BiConvGRUCell(input_channels, input_channels, 3)
        self.bidirection_conv = nn.Conv2d(2*input_channels, input_channels, 3, 1, 1)

    def forward(self, x):
        sequence_forward = []
        image = x[:,0,:,:,:]       
        #forward
        for i in range(x.shape[1]):
            image = self.convgru_forward(x[:,i,:,:,:], image) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence_forward.append(image)
       

        #backward
        image = sequence_forward[-1]   
        sequence_backward = []
        for i in range(x.shape[1]):
            image = self.convgru_backward(sequence_forward[x.shape[1]-1-i], image)
            sequence_backward.append(image)

        #连接
        sequence_backward = sequence_backward[::-1]
        sequence = []
        for i in range(x.shape[1]):
            image = torch.tanh(self.bidirection_conv(torch.cat((sequence_forward[i], sequence_backward[i]), dim=1)))
            sequence.append(image)

        last_sequence = sequence[-1] # 'b,c,h,w'
        sequences = torch.stack(sequence, dim=1)#'b,c,h,w'->'b,t,c,h,w'

        return sequences, last_sequence

class Maxpooling(nn.Module):
    def __init__(self) :
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        for i in range(x.shape[1]):
            f = x[:,i,:,:,:]            
            # 'b, 1, h, w' [64,1,64,64]
            pool = self.maxpool(f)
            pool = pool.unsqueeze(1)
            if i == 0:
                pool_result = pool              
            else:
                pool_result = torch.concat([pool_result, pool], 1)
        return pool_result

class DownConvBiRNN(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        gru_channels = out_channels
        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.birnn = BiConvGRU(gru_channels)
        self.down = Maxpooling()

    def forward(self, x):
        conv = self.doubleconv(x)
        sequences, last_result = self.birnn(conv)      
        down = self.down(sequences) 
        return last_result, down

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvCell(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvCell(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BR_UNet(nn.Module):

    def __init__(self, num_channels, num_classes, bilinear=False):
        super(BR_UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # self.convbirnn1 = DownConvBiRNN(1, 16)
        # self.convbirnn2 = DownConvBiRNN(16, 32)
        # self.convbirnn3 = DownConvBiRNN(32, 64)
        # self.convbirnn4 = DownConvBiRNN(64, 128)
        # self.convbirnn5 = DownConvBiRNN(128, 256)

        # self.up1 = Up(256, 128 // factor, bilinear)
        # self.up2 = Up(128, 64 // factor, bilinear)
        # self.up3 = Up(64, 32 // factor, bilinear)
        # self.up4 = Up(32, 16, bilinear)
        # self.outc = OutConv(16, num_classes)

        self.convbirnn1 = DownConvBiRNN(1, 32)
        self.convbirnn2 = DownConvBiRNN(32, 64)
        self.convbirnn3 = DownConvBiRNN(64, 128)
        self.convbirnn4 = DownConvBiRNN(128, 256)
        self.convbirnn5 = DownConvBiRNN(256, 512)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, num_classes)

    

    def forward(self, x):
        b, t, h, w = x.shape
        x = x.reshape(b, t, 1, h, w)
        h1, x1 = self.convbirnn1(x)
        h2, x2 = self.convbirnn2(x1)
        h3, x3 = self.convbirnn3(x2)
        h4, x4 = self.convbirnn4(x3)
        h5, _ = self.convbirnn5(x4) # b,t,c,h,w
 
        x = self.up1(h5, h4)
        x = self.up2(x, h3)
        x = self.up3(x, h2)
        x = self.up4(x, h1)
        logits = self.outc(x)

        return logits
