# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import Classifier

class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer5 = torch.nn.LSTM(2048, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)
        self.layer10 = torch.nn.LSTM(2048, 1024)
        self.layer13 = Classifier(1024, 32320)

    def forward(self, input3, input2, input0):
        out0 = input0.clone()
        out1 = None
        out2 = input2.clone()
        out3 = input3.clone()
        out4 = None
        out5 = self.layer5(out0, out1)
        out6 = out5[0]
        out6 = out6 + out2
        out8 = self.layer8(out6)
        out9 = torch.cat([out8, out3], 2)
        out10 = self.layer10(out9, out4)
        out11 = out10[0]
        out11 = out11 + out6
        out13 = self.layer13(out11)
        return out13

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
