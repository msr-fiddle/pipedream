# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer2 = torch.nn.LSTM(1024, 1024)
        self.layer5 = torch.nn.Dropout(p=0.2)
        self.layer6 = torch.nn.LSTM(1024, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)

    def forward(self, input1, input0):
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
        return (out7, out9)

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
