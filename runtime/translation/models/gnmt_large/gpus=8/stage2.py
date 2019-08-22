# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import RecurrentAttention

class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer8 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer10 = torch.nn.Dropout(p=0.2)
        self.layer11 = torch.nn.LSTM(1024, 1024)
        self.layer14 = torch.nn.Dropout(p=0.2)
        self.layer15 = torch.nn.LSTM(1024, 1024)
        self.layer18 = RecurrentAttention(1024, 1024, 1024)
        self.layer21 = torch.nn.Dropout(p=0.2)

    def forward(self, input2, input1, input0, input3):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out3 = input3.clone()
        out5 = None
        out6 = None
        out7 = out0
        out8 = self.layer8(out3)
        out7 = out7 + out1
        out10 = self.layer10(out7)
        out11 = self.layer11(out10)
        out12 = out11[0]
        out12 = out12 + out7
        out14 = self.layer14(out12)
        out15 = self.layer15(out14)
        out16 = out15[0]
        out16 = out16 + out12
        out18 = self.layer18(out8, out6, out16, out2)
        out19 = out18[2]
        out20 = out18[0]
        out21 = self.layer21(out20)
        return (out19, out21)
