# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage6(torch.nn.Module):
    def __init__(self):
        super(Stage6, self).__init__()
        self.layer5 = torch.nn.LSTM(2048, 1024)

    def forward(self, input1, input3, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = None
        out3 = input3.clone()
        out4 = torch.cat([out0, out1], 2)
        out5 = self.layer5(out4, out2)
        out6 = out5[0]
        out6 = out6 + out3
        return out6
