import torch
import torch.nn as nn
from models.utils import InitWeights
from .DRM import DRM


class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2, kernel_size=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(
                in_size, out_size, kernel_size=kernel_size, stride=kernel_size, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv3d(in_size, out_size, 1))

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet_Nested_3D(nn.Module):

    def __init__(self, num_channels=1, num_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested_3D, self).__init__()
        self.in_channels = num_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.maxpoolV2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv00 = unetConv3(
            self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv3(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv3(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv3(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(
            filters[1], filters[0], self.is_deconv, kernel_size=(1, 2, 2))
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(
            filters[1], filters[0], self.is_deconv, 3, kernel_size=(1, 2, 2))
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(
            filters[1], filters[0], self.is_deconv, 4, kernel_size=(1, 2, 2))
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(
            filters[1], filters[0], self.is_deconv, 5, kernel_size=(1, 2, 2))

        # final conv (without any concat)
        # self.final_1 = nn.Conv2d(filters[0], num_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], num_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], num_classes, 1)
        # self.final_4 = nn.Conv2d(filters[0], num_classes, 1)
        self.DRM1 = DRM(filters[0], num_classes)
        self.DRM2 = DRM(filters[0], num_classes)
        self.DRM3 = DRM(filters[0], num_classes)
        self.DRM4 = DRM(filters[0], num_classes)
        self.apply(InitWeights)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpoolV2(X_00)    # 16*256*256
        X_10 = self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.DRM1(X_01)
        final_2 = self.DRM2(X_02)
        final_3 = self.DRM3(X_03)
        final_4 = self.DRM4(X_04)

        final = (final_1+final_2+final_3+final_4)/4

        if self.is_ds:
            return final
        else:
            return final_4

# if __name__ == '__main__':
#     print('#### Test Case ###')
#     from torch.autograd import Variable
#     x = Variable(torch.rand(2,1,64,64)).cuda()
#     model = UNet_Nested().cuda()
#     param = count_param(model)
#     y = model(x)
#     print('Output shape:',y.shape)
#     print('UNet++ totoal parameters: %.2fM (%d)'%(param/1e6,param))
