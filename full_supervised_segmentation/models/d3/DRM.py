import torch.nn as nn

class DRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DRM, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x.squeeze(1))
        return x
