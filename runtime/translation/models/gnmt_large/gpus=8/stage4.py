# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.layer5 = torch.nn.LSTM(2048, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)
        self.layer10 = torch.nn.LSTM(2048, 1024)
        self.layer13 = torch.nn.Dropout(p=0.2)

    def forward(self, input3, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input3.clone()
        out3 = None
        out4 = None
        out5 = self.layer5(out0, out4)
        out6 = out5[0]
        out6 = out6 + out1
        out8 = self.layer8(out6)
        out9 = torch.cat([out8, out2], 2)
        out10 = self.layer10(out9, out3)
        out11 = out10[0]
        out11 = out11 + out6
        out13 = self.layer13(out11)
        out14 = torch.cat([out13, out2], 2)
        return (out2, out11, out14)
