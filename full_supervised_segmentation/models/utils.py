
from torch import nn


def InitWeights(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
        module.weight = nn.init.kaiming_normal_(module.weight, a=.02)
        if module.bias is not None:
            module.bias = nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
