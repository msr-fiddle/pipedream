# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage6(torch.nn.Module):
    def __init__(self):
        super(Stage6, self).__init__()
        self.layer4 = torch.nn.Dropout(p=0.2)
        self.layer6 = torch.nn.LSTM(2048, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = None
        out3 = out0
        out4 = self.layer4(out1)
        out5 = torch.cat([out4, out3], 2)
        out6 = self.layer6(out5, out2)
        out7 = out6[0]
        out8 = self.layer8(out7)
        return (out3, out7, out8)

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
