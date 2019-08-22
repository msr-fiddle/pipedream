# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.encoder import EmuBidirLSTM

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer4 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer5 = EmuBidirLSTM(1024, 1024)
        self.layer6 = torch.nn.Dropout(p=0.2)
        self.layer7 = torch.nn.LSTM(2048, 1024)
        self.layer9 = torch.nn.Dropout(p=0.2)

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out4 = self.layer4(out0)
        out5 = self.layer5(out4, out1)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = out7[0]
        out9 = self.layer9(out8)
        return (out9, out8)

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
