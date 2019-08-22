# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer4 = torch.nn.Dropout(p=0.5)
        self.layer5 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Dropout(p=0.5)
        self.layer8 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = out1.size(0)
        out3 = out1.view(out2, 9216)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        return out10
