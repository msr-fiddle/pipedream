# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = torch.nn.LSTM(1024, 1024)
        self.layer5 = torch.nn.Dropout(p=0.2)
        self.layer6 = torch.nn.LSTM(1024, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)
        self.layer10 = torch.nn.LSTM(1024, 1024)
        self.layer13 = torch.nn.Dropout(p=0.2)
        self.layer14 = torch.nn.LSTM(1024, 1024)

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = self.layer2(out0)
        out3 = out2[0]
        out3 = out3 + out1
        out5 = self.layer5(out3)
        out6 = self.layer6(out5)
        out7 = out6[0]
        out7 = out7 + out3
        out9 = self.layer9(out7)
        out10 = self.layer10(out9)
        out11 = out10[0]
        out11 = out11 + out7
        out13 = self.layer13(out11)
        out14 = self.layer14(out13)
        return (out11, out14[0])
