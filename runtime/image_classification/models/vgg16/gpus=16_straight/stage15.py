# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage15(torch.nn.Module):
    def __init__(self):
        super(Stage15, self).__init__()
        self.layer1 = torch.nn.ReLU(inplace=True)
        self.layer2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer5 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Dropout(p=0.5)
        self.layer8 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Dropout(p=0.5)
        self.layer11 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = out2.size(0)
        out4 = out2.view(out3, -1)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return out11

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
