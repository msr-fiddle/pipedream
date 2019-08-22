# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = torch.nn.ReLU(inplace=True)
        self.layer8 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer9 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10 = torch.nn.ReLU(inplace=True)
        self.layer11 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self._initialize_weights()

    def forward(self, input1, input0):
        out0 = input1.clone()
        out1 = input0.clone()
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out5 += out0
        out7 = self.layer7(out5)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return (out7, out11)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
