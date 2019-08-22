# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.layer6 = torch.nn.LSTM(2048, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)
        self.layer11 = torch.nn.LSTM(2048, 1024)
        self.layer14 = torch.nn.Dropout(p=0.2)

    def forward(self, input3, input2, input0):
        out0 = input0.clone()
        out1 = None
        out2 = input2.clone()
        out3 = input3.clone()
        out4 = None
        out5 = None
        out6 = self.layer6(out0, out1)
        out7 = out6[0]
        out7 = out7 + out2
        out9 = self.layer9(out7)
        out10 = torch.cat([out9, out3], 2)
        out11 = self.layer11(out10, out5)
        out12 = out11[0]
        out12 = out12 + out7
        out14 = self.layer14(out12)
        return (out3, out12, out14)
