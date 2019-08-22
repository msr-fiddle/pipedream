# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer7 = torch.nn.Dropout(p=0.2)
        self.layer9 = torch.nn.LSTM(2048, 1024)
        self.layer11 = torch.nn.Dropout(p=0.2)
        self.layer13 = torch.nn.LSTM(2048, 1024)
        self.layer16 = torch.nn.Dropout(p=0.2)

    def forward(self, input0, input1):
        input4 = [None, None, None, None]
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input4
        out4 = out2[2]
        out5 = out0
        out6 = out2[1]
        out7 = self.layer7(out1)
        out8 = torch.cat([out7, out5], 2)
        out9 = self.layer9(out8, out6)
        out10 = out9[0]
        out11 = self.layer11(out10)
        out12 = torch.cat([out11, out5], 2)
        out13 = self.layer13(out12, out4)
        out14 = out13[0]
        out14 = out14 + out10
        out16 = self.layer16(out14)
        return (out5, out14, out16)

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
