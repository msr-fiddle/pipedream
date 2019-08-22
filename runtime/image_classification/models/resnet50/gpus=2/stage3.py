# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12 = torch.nn.ReLU(inplace=True)
        self.layer13 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer14 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer16 = torch.nn.ReLU(inplace=True)
        self.layer17 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.layer20 = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)

        self._initialize_weights()

    def forward(self, input0, input1):
        out0 = input1.clone()
        out1 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 += out1
        out6 = self.layer6(out4)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out14 += out6
        out16 = self.layer16(out14)
        out17 = self.layer17(out16)
        out18 = out17.size(0)
        out19 = out17.view(out18, -1)
        out20 = self.layer20(out19)
        return out20

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
