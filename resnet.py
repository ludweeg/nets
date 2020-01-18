import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                           padding=dilation, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        print(x.shape)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

bb = BasicBlock(5, 5, norm_layer=nn.BatchNorm2d)
x = torch.randn((1, 5, 5, 5))
out = bb(x)
