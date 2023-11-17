import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dp=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),

            nn.InstanceNorm2d(mid_channels),
            nn.Dropout2d(dp),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.Dropout2d(dp),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvGRUCell(nn.Module):
    """
        ICLR2016: Delving Deeper into Convolutional Networks for Learning Video Representations
        url: https://arxiv.org/abs/1511.06432
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, cuda_flag=True):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.cuda_flag = cuda_flag
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        padding = self.kernel_size // 2
        self.reset_gate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.output_gate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        # init
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        if hidden is None:
            size_h = [x.data.size()[0], self.hidden_channels] + \
                list(x.data.size()[2:])
            if self.cuda_flag:
                hidden = torch.zeros(size_h).cuda()
            else:
                hidden = torch.zeros(size_h)

        inputs = torch.cat((x, hidden), dim=1)
        reset_gate = torch.sigmoid(self.reset_gate(inputs))
        update_gate = torch.sigmoid(self.update_gate(inputs))

        reset_hidden = reset_gate * hidden
        reset_inputs = torch.tanh(self.output_gate(
            torch.cat((x, reset_hidden), dim=1)))
        new_hidden = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden


class ConvGRU(nn.Module):
    def __init__(self, input_channels, output_channels, is_down=True):
        super().__init__()
        self.is_down = is_down
        self.down = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.doubel_Conv = DoubleConv(input_channels, output_channels)
        self.convgru_forward = ConvGRUCell(output_channels, output_channels, 3)

    def forward(self, x):

       

        sequence = []
        for i in range(x.shape[2]):
            # 'b,t,c,h,w' -> 'b,c,h,w'
            image = self.doubel_Conv(x[:, :, i, :, :])
            sequence.append(image)
        x = torch.stack(sequence, dim=2)

        image = x[:, :, 0, :, :]

        # forward
        for i in range(x.shape[2]):
            # 'b,t,c,h,w' -> 'b,c,h,w'
            image = self.convgru_forward(x[:,  :,i, :, :], image)

        down = self.down(x) if self.is_down else None

        return down, image


class Up(nn.Module):
    def __init__(self, out_c, dp=0):
        super(Up, self).__init__()

        self.up = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(out_c*2, out_c, kernel_size=1)

        )

        self.conv = DoubleConv(out_c*2, out_c, dp=dp)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class ST_UNet(nn.Module):
    def __init__(self, num_channels, num_classes, dp=0.2):
        super(ST_UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.en1 = ConvGRU(num_channels, 64, True)
        self.en2 = ConvGRU(64, 128, True)
        self.en3 = ConvGRU(128, 256, True)
        self.en4 = ConvGRU(256, 512, True)
        self.en5 = ConvGRU(512, 1024, False)

        self.de1 = Up(64)
        self.de2 = Up(128)
        self.de3 = Up(256)
        self.de4 = Up(512)

        self.outc = OutConv(64, num_classes)

    def forward(self, x):

        x, sc1 = self.en1(x)
        x, sc2 = self.en2(x)
        x, sc3 = self.en3(x)
        x, sc4 = self.en4(x)
        _, sc5 = self.en5(x)
        x = self.de4(sc5, sc4)
        x = self.de3(x, sc3)
        x = self.de2(x, sc2)
        x = self.de1(x, sc1)
        logits = self.outc(x)
        return logits
