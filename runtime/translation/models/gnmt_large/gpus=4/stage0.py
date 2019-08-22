# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.encoder import EmuBidirLSTM

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer6 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer7 = EmuBidirLSTM(1024, 1024)
        self.layer8 = torch.nn.Dropout(p=0.2)
        self.layer9 = torch.nn.LSTM(2048, 1024)
        self.layer11 = torch.nn.Dropout(p=0.2)
        self.layer12 = torch.nn.LSTM(1024, 1024)
        self.layer15 = torch.nn.Dropout(p=0.2)
        self.layer16 = torch.nn.LSTM(1024, 1024)
        self.layer19 = torch.nn.Dropout(p=0.2)
        self.layer20 = torch.nn.LSTM(1024, 1024)

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out6 = self.layer6(out0)
        out7 = self.layer7(out6, out1)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = out9[0]
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = out12[0]
        out13 = out13 + out10
        out15 = self.layer15(out13)
        out16 = self.layer16(out15)
        out17 = out16[0]
        out17 = out17 + out13
        out19 = self.layer19(out17)
        out20 = self.layer20(out19)
        out21 = out20[0]
        return (out17, out21)

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
