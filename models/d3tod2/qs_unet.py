import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,dp = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Seq_doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,dp = 0):
        super().__init__()
        self.SeqConv = DoubleConv(in_channels, out_channels, mid_channels,dp = dp)
    def forward(self, x):
        sequence = []
        for i in range(x.shape[2]):
            image = self.SeqConv(x[:,:,i,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class Sequence_down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Sequence_down, self).__init__()
        self.down = Down(in_c, out_c)

    def forward(self, x):
        sequence = []
        for i in range(x.shape[1]):
            image = self.down(x[:,i,:,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences


class Up(nn.Module):
    def __init__(self, in_c, out_c,dp=0):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
            
            )
        self.conv = DoubleConv(out_c*2,out_c,dp=dp)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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
        new_hidden   = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden

class BiConvGRU(nn.Module):
    def __init__(self, input_channels,output_channels, is_down=True ) :
        super().__init__()
        
        self.input_channels = input_channels
        self.down = Sequence_down(input_channels*2,output_channels) if is_down else None
        self.convgru_forward = BiConvGRUCell(output_channels, output_channels, 3)
        self.convgru_backward = BiConvGRUCell(output_channels, output_channels, 3)
        self.bidirection_conv = nn.Conv2d(2*output_channels, output_channels, 3, 1, 1)

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
        image = x[:,-1,:,:,:]   
        sequence_backward = []
        for i in range(x.shape[1]):
            image = self.convgru_backward(x[:,x.shape[1]-1-i,:,:,:], image)
            sequence_backward.append(image)
        # image = sequence_forward[-1]   
        # sequence_backward = []
        # for i in range(x.shape[1]):
        #     image = self.convgru_backward(sequence_forward[x.shape[1]-1-i], image)
        #     sequence_backward.append(image)

        #连接
        sequence_backward = sequence_backward[::-1]
        sequence = []
        for i in range(x.shape[1]):
            image = torch.tanh(self.bidirection_conv(torch.cat((sequence_forward[i], sequence_backward[i]), dim=1)))
            sequence.append(image)

        
        sequences = torch.stack(sequence, dim=1)#'b,c,h,w'->'b,t,c,h,w'
        last_sequence = torch.max(sequences, dim=1)[0]
        # last_sequence = torch.squeeze(self.GAP(sequences),1)

        return sequences, last_sequence


class QS_UNet(nn.Module):
    def __init__(self, num_channels, num_classes, dp=0.2):
        super(QS_UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.inc = Seq_doubleConv(num_channels, 32,dp=dp)
        self.en1 = BiConvGRU(32,32,is_down=False)
        self.en2 = BiConvGRU(32,64)
        self.en3 = BiConvGRU(64,128)
        self.en4 = BiConvGRU(128,256)
        self.en5 = BiConvGRU(256,512)

        self.up4 = Up(512, 256,dp=dp)
        self.up3 = Up(256, 128,dp=dp)
        self.up2 = Up(128, 64,dp=dp)
        self.up1 = Up(64, 32,dp=dp)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        s = x.permute(0,2, 1, 3,4) 
        x = self.inc(x)
        x,sc1 = self.en1(x)
        x,sc2 = self.en2(x,s)
        x,sc3 = self.en3(x,s)
        x,sc4 = self.en4(x,s)
        _,sc5 = self.en5(x,s)
        x = self.up4(sc5, sc4)
        x = self.up3(x, sc3)
        x = self.up2(x, sc2)
        x = self.up1(x, sc1)
        logits = self.outc(x)
        return logits


def resize_with_GPU(x, output_size,mode='trilinear',align_corners=False):

    x = F.interpolate(x, size=output_size,
                      mode='trilinear', align_corners=False)


