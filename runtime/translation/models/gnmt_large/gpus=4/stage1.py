# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import RecurrentAttention

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer10 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer12 = torch.nn.Dropout(p=0.2)
        self.layer13 = torch.nn.LSTM(1024, 1024)
        self.layer16 = torch.nn.Dropout(p=0.2)
        self.layer17 = torch.nn.LSTM(1024, 1024)
        self.layer20 = torch.nn.Dropout(p=0.2)
        self.layer21 = torch.nn.LSTM(1024, 1024)
        self.layer24 = RecurrentAttention(1024, 1024, 1024)
        self.layer27 = torch.nn.Dropout(p=0.2)
        self.layer29 = torch.nn.LSTM(2048, 1024)

    def forward(self, input1, input2, input0, input3):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out3 = input3.clone()
        out5 = None
        out6 = None
        out7 = None
        out8 = None
        out9 = None
        out10 = self.layer10(out3)
        out0 = out0 + out1
        out12 = self.layer12(out0)
        out13 = self.layer13(out12)
        out14 = out13[0]
        out14 = out14 + out0
        out16 = self.layer16(out14)
        out17 = self.layer17(out16)
        out18 = out17[0]
        out18 = out18 + out14
        out20 = self.layer20(out18)
        out21 = self.layer21(out20)
        out22 = out21[0]
        out22 = out22 + out18
        out24 = self.layer24(out10, out9, out22, out2)
        out25 = out24[2]
        out26 = out24[0]
        out27 = self.layer27(out26)
        out28 = torch.cat([out27, out25], 2)
        out29 = self.layer29(out28, out8)
        out30 = out29[0]
        return (out25, out30)

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
