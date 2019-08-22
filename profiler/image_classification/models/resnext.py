# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
import math
from .resnet import ResNet
from collections import OrderedDict

class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, cardinality=32, baseWidth=4):
        """ Constructor
        Args:
            in_planes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()

        width = int(math.floor((planes * cardinality * baseWidth) / 64))
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(ResNet):
    def __init__(self, block, layers, num_classes=1000, cardinality=32, baseWidth=4, shortcut='C'):
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.shortcut = shortcut
        super(ResNeXt, self).__init__(block, layers, num_classes=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        reshape = stride != 1 or self.in_planes != planes * block.expansion
        useConv = (self.shortcut == 'C') or (self.shortcut == 'B' and reshape)

        if useConv:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif reshape:
            downsample = nn.AvgPool2d(3, stride=stride)

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.cardinality, self.baseWidth))
        self.in_planes = planes * block.expansion

        shortcut = None
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, shortcut, self.cardinality, self.baseWidth))

        return nn.Sequential(*layers)

def resnext18(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-18 model.
    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneck, [2, 2, 2, 2], cardinality=cardinality,
                    baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model

def resnext50(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], cardinality=cardinality,
                    baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model


def resnext101(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], cardinality=cardinality,
                    baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model


def resnext152(cardinality=32, baseWidth=4, shortcut='C', **kwargs):
    """Constructs a ResNeXt-152 model.
    Args:
        cardinality (int): Cardinality of the aggregated transform
        baseWidth (int): Base width of the grouped convolution
        shortcut ('A'|'B'|'C'): 'B' use 1x1 conv to downsample, 'C' use 1x1 conv on every residual connection
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], cardinality=cardinality,
                    baseWidth=baseWidth, shortcut=shortcut, **kwargs)
    return model
