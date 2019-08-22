# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
ResNet implementation heavily inspired by the torchvision ResNet implementation
(needed for ResNeXt model implementation)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BasicBlockVGG(nn.Module):
    expansion = 1
    """A BasicBlock without the residual connections
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """

    def __init__(self, in_planes, planes, stride=1, option='cifar10'):
        super(BasicBlockVGG, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

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

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckVGG(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(BottleneckVGG, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes= 1000):
        super(ResNet, self).__init__()

        self.in_planes = 64

        ip = self.in_planes
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(3, ip, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, ip, layers[0], stride=1)
        self.layer2 = self._make_layer(block, ip * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ip * 4, layers[2], stride=2)

        self.layer4 = self._make_layer(block, ip * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)

        #Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            if i == 0: self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, get_features = False):
        if get_features: features = OrderedDict()

        out = self.relu(self.bn1(self.conv1(x)))
        if self.layer4: out = self.maxpool(out)
        if get_features:
            features[0] = out.detach()

        out = self.layer1(out)
        if get_features:
            features[1] = out.detach()

        out = self.layer2(out)
        if get_features:
            features[2] = out.detach()

        out = self.layer3(out)
        if get_features:
            features[3] = out.detach()

        if self.layer4:
            out = self.layer4(out)
            if get_features:
                features[4] = out.detach()
            out = self.avgpool(out)
        else:
            avgpool_module = nn.AvgPool2d(out.size()[3])
            out = avgpool_module(out)
        if get_features:
            return features
        out = out.view(out.size(0), -1)
        # Fully connected layer to get to the class
        out = self.linear(out)
        return out
