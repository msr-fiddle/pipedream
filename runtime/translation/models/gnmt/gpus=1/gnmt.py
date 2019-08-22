# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.encoder import EmuBidirLSTM
from seq2seq.models.decoder import RecurrentAttention
from seq2seq.models.decoder import Classifier

class GNMTGenerated(torch.nn.Module):
    def __init__(self):
        super(GNMTGenerated, self).__init__()
        self.layer11 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer12 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer13 = EmuBidirLSTM(1024, 1024)
        self.layer14 = torch.nn.Dropout(p=0.2)
        self.layer15 = torch.nn.LSTM(2048, 1024)
        self.layer17 = torch.nn.Dropout(p=0.2)
        self.layer18 = torch.nn.LSTM(1024, 1024)
        self.layer21 = torch.nn.Dropout(p=0.2)
        self.layer22 = torch.nn.LSTM(1024, 1024)
        self.layer25 = RecurrentAttention(1024, 1024, 1024)
        self.layer28 = torch.nn.Dropout(p=0.2)
        self.layer30 = torch.nn.LSTM(2048, 1024)
        self.layer32 = torch.nn.Dropout(p=0.2)
        self.layer34 = torch.nn.LSTM(2048, 1024)
        self.layer37 = torch.nn.Dropout(p=0.2)
        self.layer39 = torch.nn.LSTM(2048, 1024)
        self.layer42 = Classifier(1024, 32320)

    def forward(self, input0, input1, input2):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out3 = [None, None, None, None]  # out3 is hidden, might need to be initialized differently.
        out4 = out3[3]
        out5 = out3[2]
        out6 = out3[1]
        out7 = out3[0]
        out11 = self.layer11(out2)
        out12 = self.layer12(out0)
        out13 = self.layer13(out12, out1)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = out15[0]
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = out18[0]
        out19 = out19 + out16
        out21 = self.layer21(out19)
        out22 = self.layer22(out21)
        out23 = out22[0]
        out23 = out23 + out19
        out25 = self.layer25(out11, out7, out23, out1)
        out26 = out25[2]
        out27 = out25[0]
        out28 = self.layer28(out27)
        out29 = torch.cat([out28, out26], 2)
        out30 = self.layer30(out29, out6)
        out31 = out30[0]
        out32 = self.layer32(out31)
        out33 = torch.cat([out32, out26], 2)
        out34 = self.layer34(out33, out5)
        out35 = out34[0]
        out35 = out35 + out31
        out37 = self.layer37(out35)
        out38 = torch.cat([out37, out26], 2)
        out39 = self.layer39(out38, out4)
        out40 = out39[0]
        out40 = out40 + out35
        out42 = self.layer42(out40)
        return out42

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
