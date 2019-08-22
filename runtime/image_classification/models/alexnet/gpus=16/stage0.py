# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer5 = torch.nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer8 = torch.nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer17 = torch.nn.Dropout(p=0.5)

    

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = out14.size(0)
        out16 = out14.view(out15, 9216)
        out17 = self.layer17(out16)
        return out17
