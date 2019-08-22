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

    def forward(self, input0, input1):
        out0 = input0.clone()
        out1 = input1.clone()
        out6 = self.layer6(out0)
        out7 = self.layer7(out6, out1)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = out9[0]
        out11 = self.layer11(out10)
        return (out11, out10)
