# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4
