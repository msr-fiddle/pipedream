# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer6 = torch.nn.LSTM(2048, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)
        self.layer10 = torch.nn.LSTM(2048, 1024)
        self.layer13 = torch.nn.Dropout(p=0.2)

    def forward(self, input4, input0):
        out0 = input0.clone()
        out1 = input4.clone()
        out2 = None
        out3 = None
        out4 = None
        out5 = torch.cat([out0, out1], 2)
        out6 = self.layer6(out5, out2)
        out7 = out6[0]
        out8 = self.layer8(out7)
        out9 = torch.cat([out8, out1], 2)
        out10 = self.layer10(out9, out4)
        out11 = out10[0]
        out11 = out11 + out7
        out13 = self.layer13(out11)
        out14 = torch.cat([out13, out1], 2)
        return (out1, out11, out14)
