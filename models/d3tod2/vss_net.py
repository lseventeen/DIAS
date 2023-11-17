import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import InitWeights

class Res_conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN = True):
        super(Res_conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN = True):
        super(Conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
       

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)
        out = self.conv(x)
      
        return out

    
class Seq_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dp = 0,is_BN = True):
        super().__init__()
        self.SeqConv = Res_conv(in_channels, out_channels,dp = dp,is_BN = is_BN)
    def forward(self, x):
        sequence = []
        for i in range(x.shape[2]):
            image = self.SeqConv(x[:,:,i,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences


class Down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class Sequence_down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Sequence_down, self).__init__()
        self.down = Down(in_c, out_c,dp,is_BN =is_BN)

    def forward(self, x):
        sequence = []
        for i in range(x.shape[1]):
            image = self.down(x[:,i,:,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences





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
        # for m in self.state_dict():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)

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
        new_hidden   = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden

class BiConvGRU(nn.Module):
    def __init__(self, input_channels,output_channels, is_down=True,dp=0.2,is_BN = True) :
        super().__init__()
        
        self.input_channels = input_channels
        self.down = Sequence_down(input_channels*2,output_channels,dp,is_BN = is_BN) if is_down else None
        self.convgru_forward = BiConvGRUCell(output_channels, output_channels, 3)
        self.convgru_backward = BiConvGRUCell(output_channels, output_channels, 3)
        self.bidirection_conv = nn.Conv2d(2*output_channels, output_channels, 3, 1, 1)
        self.fuse_conv = nn.Conv2d(2*output_channels, output_channels, 1)

    def forward(self, x, s = None):
        if self.down is not None:
            B, S, _, H, W = x.shape
            s = F.interpolate(s, size=(self.input_channels, H, W),
                        mode='trilinear', align_corners=False)
            
            x = self.down(torch.cat([s, x], dim=2))
        
        sequence_forward = []
        image = x[:,0,:,:,:]       
        #forward
        for i in range(x.shape[1]):
            image = self.convgru_forward(x[:,i,:,:,:], image) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence_forward.append(image)
       

        # backward
        # image = x[:,-1,:,:,:]   
        # sequence_backward = []
        # for i in range(x.shape[1]):
        #     image = self.convgru_backward(x[:,x.shape[1]-1-i,:,:,:], image)
        #     sequence_backward.append(image)
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
            # image = self.bidirection_conv(torch.cat((sequence_forward[i], sequence_backward[i]), dim=1))
            sequence.append(image)


        # sequences = torch.stack(sequence_backward, dim=1)#'b,c,h,w'->'b,t,c,h,w'
        # sequences = torch.stack(sequence_forward, dim=1)#'b,c,h,w'->'b,t,c,h,w'
        sequences = torch.stack(sequence, dim=1)#'b,c,h,w'->'b,t,c,h,w'4
        # last_sequence = self.fuse_conv(torch.cat([sequence_forward[-1], sequence_backward[-1]], dim=1))
        last_sequence = torch.max(sequences, dim=1)[0]



        
    
       

        return sequences, last_sequence


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c,is_BN = True):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        # self.norm = nn.BatchNorm2d(out_c)
        self.norm =nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c)
        # self.norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class Up(nn.Module):
    def __init__(self, in_c, out_c, dp=0,is_BN = True):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x



class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False,is_BN = True):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c,is_BN = is_BN)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = Res_conv(out_c, out_c, dp,is_BN= is_BN )
        if self.is_up == True:
            self.up = Up(out_c, out_c//2, dp,is_BN= is_BN)
        if self.is_down == True:
            self.down = Down(out_c, out_c*2,dp,is_BN= is_BN)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class VSS_Net(nn.Module):
    def __init__(self, num_classes=3, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True):
        super(VSS_Net, self).__init__()
        
        self.num_channels = num_channels

        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]


        self.inc = Seq_conv(num_channels, filters[0],dp=dropout,is_BN=False)
        self.en1 = BiConvGRU(filters[0],filters[0],is_down=False,is_BN=False)
        self.en2 = BiConvGRU(filters[0],filters[1],is_BN=False)
        self.en3 = BiConvGRU(filters[1],filters[2],is_BN=False)
        self.en4 = BiConvGRU(filters[2],filters[3],is_BN=False)



        # self.block1_3 = block(
        #     self.num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights)

    def forward(self, x):
        s = x.permute(0,2, 1, 3,4) 
        x = self.inc(x)
        x,sc1 = self.en1(x)
        x,sc2 = self.en2(x,s)
        x,sc3 = self.en3(x,s)
        _,sc4 = self.en4(x,s)

        # x1_3, x_down1_3 = self.block1_3(sc1)
        x1_2, x_down1_2 = self.block1_2(sc1)
        x2_2, x_up2_2 = self.block2_2(sc2)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1 = self.block3_1(sc3)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(sc4)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                      self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)
        # output = torch.softmax(output, dim=1)

        return output



