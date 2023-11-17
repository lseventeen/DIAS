import torch
import torch.nn as nn
from models.utils import InitWeights


class Conv3d(nn.Module):
    def __init__(self, in_c, out_c,dropout_p = 0):
        super(Conv3d, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=(1,2,2), padding=0,stride =(1,2,2), bias=False)
        self.batcnorn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        if dropout_p != 0:
            self.dropout = nn.Dropout3d(dropout_p)
        else:
            self.dropout = None
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        return self.relu(self.batcnorn(x))
    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c, dropout_p = 0, is_first_stage = False, post_conv = True):
        super(Encoder, self).__init__()
        if is_first_stage:
            self.conv = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0)
        else:
            self.conv = Conv3d(in_c, out_c, dropout_p)
        if post_conv:
            self.post_conv = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        else:
            self.post_conv = None
    def forward(self, x):
        
        x = self.conv(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x
    
class Upsampling(nn.Module):
    def __init__(self, in_c, out_c):
        super(Upsampling, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,stride =1, bias=False)
        self.batcnorn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
       
       
    def forward(self, x):
       
        return self.relu(self.batcnorn(self.conv(self.upsampling(x))))

class Channel_attention_block(nn.Module):
    def __init__(self, in_c):
        super(Channel_attention_block, self).__init__()
        self.GAP = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_c*2,in_c, kernel_size=1, padding=0,stride =1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_c,in_c, kernel_size=1, padding=0,stride =1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,lf,hf):
        x = torch.cat((lf,hf), 1)
        x =  self.sigmoid(self.conv2(self.relu(self.conv1(self.GAP(x)))))
        return torch.mul(lf,x)+hf
     
        


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(Decoder, self).__init__()
        self.upsampling = Upsampling(in_c, out_c)
        self.CAB = Channel_attention_block(out_c)
        self.conv = nn.Conv2d(out_c,out_c, kernel_size=3, padding=1,stride =1)
    def forward(self, x,lf):



        hf = self.upsampling(x)
        x = self.CAB(torch.squeeze(lf,2),hf)
        return self.conv(x)


class SVS_Net(nn.Module):
    def __init__(self, num_channels,num_classes):
        super(SVS_Net, self).__init__()
        self.encoder1 = Encoder(num_channels,8,is_first_stage=True)
        self.encoder2 = Encoder(8,16)
        self.encoder3 = Encoder(16,32)
        self.encoder4 = Encoder(32,64)
        self.encoder5 = Encoder(64,128)
        self.encoder6 = Encoder(128,256,dropout_p=0.5)
        self.encoder7 = Encoder(256,512,dropout_p=0.5,post_conv=False)

        self.skip_connect1 = nn.Conv3d(8, 8, kernel_size=(8,1,1))
        self.skip_connect2 = nn.Conv3d(16, 16, kernel_size=(8,1,1))
        self.skip_connect3 = nn.Conv3d(32, 32, kernel_size=(8,1,1))
        self.skip_connect4 = nn.Conv3d(64, 64, kernel_size=(8,1,1))
        self.skip_connect5 = nn.Conv3d(128, 128, kernel_size=(8,1,1))
        self.skip_connect6 = nn.Conv3d(256, 256, kernel_size=(8,1,1))
        self.skip_connect7 = nn.Conv3d(512, 512, kernel_size=(8,1,1))

        self.decoder1 = Decoder(16,8)
        self.decoder2 = Decoder(32,16)
        self.decoder3 = Decoder(64,32)
        self.decoder4 = Decoder(128,64)
        self.decoder5 = Decoder(256,128)
        self.decoder6 = Decoder(512,256)

        self.conv = nn.Conv2d(8,num_classes, kernel_size=1)
    def forward(self, x):
        en1=self.encoder1(x)
        en2=self.encoder2(en1)
        en3=self.encoder3(en2)
        en4=self.encoder4(en3)
        en5=self.encoder5(en4)
        en6=self.encoder6(en5)
        en7=self.encoder7(en6)

        sc1 = self.skip_connect1(en1)
        sc2 = self.skip_connect2(en2)
        sc3 = self.skip_connect3(en3)
        sc4 = self.skip_connect4(en4)
        sc5 = self.skip_connect5(en5)
        sc6 = self.skip_connect6(en6)
        sc7 = self.skip_connect7(en7)
        
        x = self.decoder6(torch.squeeze(sc7,2),sc6)
        x = self.decoder5(x,sc5)
        x = self.decoder4(x,sc4)
        x = self.decoder3(x,sc3)
        x = self.decoder2(x,sc2)
        x = self.decoder1(x,sc1)

        x = self.conv(x)

        return x


        
        
        


