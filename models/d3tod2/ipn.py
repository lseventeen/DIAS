import torch.nn as nn
from models.utils import InitWeights
import torch.nn.functional as F


class PLM(nn.Module):
    def __init__(self, in_c, out_c):
        super(PLM, self).__init__()
        self.conv = nn.Sequential(

            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

        )

        self.UN_maxpooling = nn.MaxPool3d(
            kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):

        return self.UN_maxpooling(self.conv(x))


class IPN(nn.Module):

    def __init__(self, num_channels=1, num_classes=1, feature_scale=4):
        super(IPN, self).__init__()
        self.in_channels = num_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.PLM1 = PLM(1, filters[0])
        self.PLM2 = PLM(filters[0], filters[1])
        self.PLM3 = PLM(filters[1], filters[2])
        self.PLM4 = PLM(filters[2], filters[3])
        self.PLM5 = PLM(filters[3], filters[4])
        self.RC_conv = nn.Sequential(

            nn.Conv3d(filters[4], filters[2],
                      kernel_size=3, padding=1, bias=False),

            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(filters[2], filters[0],
                      kernel_size=3, padding=1, bias=False),

            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(filters[0], num_classes,
                      kernel_size=3, padding=1, bias=False),

            nn.LeakyReLU(0.1, inplace=True),

        )

        self.apply(InitWeights)

    def forward(self, x):
        x = F.interpolate(x, size=[32, 64, 64], mode='trilinear',
                          align_corners=False)
        x = self.PLM1(x)
        x = self.PLM2(x)
        x = self.PLM3(x)
        x = self.PLM4(x)
        x = self.PLM5(x)
        x = self.RC_conv(x)
        final = x.squeeze(2)

        return final
