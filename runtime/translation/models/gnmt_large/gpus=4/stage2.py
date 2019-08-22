# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer9 = torch.nn.Dropout(p=0.2)
        self.layer11 = torch.nn.LSTM(2048, 1024)
        self.layer14 = torch.nn.Dropout(p=0.2)
        self.layer16 = torch.nn.LSTM(2048, 1024)
        self.layer19 = torch.nn.Dropout(p=0.2)
        self.layer21 = torch.nn.LSTM(2048, 1024)
        self.layer24 = torch.nn.Dropout(p=0.2)
        self.layer26 = torch.nn.LSTM(2048, 1024)
        self.layer29 = torch.nn.Dropout(p=0.2)

    def forward(self, input8, input0):
        out0 = input0.clone()
        out1 = input8.clone()
        out2 = None
        out3 = None
        out4 = None
        out7 = None
        out8 = out0
        out9 = self.layer9(out8)
        out10 = torch.cat([out9, out1], 2)
        out11 = self.layer11(out10, out2)
        out12 = out11[0]
        out12 = out12 + out8
        out14 = self.layer14(out12)
        out15 = torch.cat([out14, out1], 2)
        out16 = self.layer16(out15, out3)
        out17 = out16[0]
        out17 = out17 + out12
        out19 = self.layer19(out17)
        out20 = torch.cat([out19, out1], 2)
        out21 = self.layer21(out20, out4)
        out22 = out21[0]
        out22 = out22 + out17
        out24 = self.layer24(out22)
        out25 = torch.cat([out24, out1], 2)
        out26 = self.layer26(out25, out7)
        out27 = out26[0]
        out27 = out27 + out22
        out29 = self.layer29(out27)
        out30 = torch.cat([out29, out1], 2)
        return (out27, out30, out1)

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
