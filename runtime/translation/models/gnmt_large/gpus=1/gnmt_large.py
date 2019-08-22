# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.encoder import EmuBidirLSTM
from seq2seq.models.decoder import RecurrentAttention
from seq2seq.models.decoder import Classifier

class GNMTGenerated(torch.nn.Module):
    def __init__(self):
        super(GNMTGenerated, self).__init__()
        self.layer15 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer16 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer17 = EmuBidirLSTM(1024, 1024)
        self.layer18 = torch.nn.Dropout(p=0.2)
        self.layer19 = torch.nn.LSTM(2048, 1024)
        self.layer21 = torch.nn.Dropout(p=0.2)
        self.layer22 = torch.nn.LSTM(1024, 1024)
        self.layer25 = torch.nn.Dropout(p=0.2)
        self.layer26 = torch.nn.LSTM(1024, 1024)
        self.layer29 = torch.nn.Dropout(p=0.2)
        self.layer30 = torch.nn.LSTM(1024, 1024)
        self.layer33 = torch.nn.Dropout(p=0.2)
        self.layer34 = torch.nn.LSTM(1024, 1024)
        self.layer37 = torch.nn.Dropout(p=0.2)
        self.layer38 = torch.nn.LSTM(1024, 1024)
        self.layer41 = torch.nn.Dropout(p=0.2)
        self.layer42 = torch.nn.LSTM(1024, 1024)
        self.layer45 = RecurrentAttention(1024, 1024, 1024)
        self.layer48 = torch.nn.Dropout(p=0.2)
        self.layer50 = torch.nn.LSTM(2048, 1024)
        self.layer52 = torch.nn.Dropout(p=0.2)
        self.layer54 = torch.nn.LSTM(2048, 1024)
        self.layer57 = torch.nn.Dropout(p=0.2)
        self.layer59 = torch.nn.LSTM(2048, 1024)
        self.layer62 = torch.nn.Dropout(p=0.2)
        self.layer64 = torch.nn.LSTM(2048, 1024)
        self.layer67 = torch.nn.Dropout(p=0.2)
        self.layer69 = torch.nn.LSTM(2048, 1024)
        self.layer72 = torch.nn.Dropout(p=0.2)
        self.layer74 = torch.nn.LSTM(2048, 1024)
        self.layer77 = torch.nn.Dropout(p=0.2)
        self.layer79 = torch.nn.LSTM(2048, 1024)
        self.layer82 = Classifier(1024, 32320)

    def forward(self, input0, input1, input2):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out4 = None
        out5 = None
        out6 = None
        out7 = None
        out8 = None
        out9 = None
        out10 = None
        out11 = None
        out15 = self.layer15(out2)
        out16 = self.layer16(out0)
        out17 = self.layer17(out16, out1)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = out19[0]
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = out22[0]
        out23 = out23 + out20
        out25 = self.layer25(out23)
        out26 = self.layer26(out25)
        out27 = out26[0]
        out27 = out27 + out23
        out29 = self.layer29(out27)
        out30 = self.layer30(out29)
        out31 = out30[0]
        out31 = out31 + out27
        out33 = self.layer33(out31)
        out34 = self.layer34(out33)
        out35 = out34[0]
        out35 = out35 + out31
        out37 = self.layer37(out35)
        out38 = self.layer38(out37)
        out39 = out38[0]
        out39 = out39 + out35
        out41 = self.layer41(out39)
        out42 = self.layer42(out41)
        out43 = out42[0]
        out43 = out43 + out39
        out45 = self.layer45(out15, out11, out43, out1)
        out46 = out45[2]
        out47 = out45[0]
        out48 = self.layer48(out47)
        out49 = torch.cat([out48, out46], 2)
        out50 = self.layer50(out49, out10)
        out51 = out50[0]
        out52 = self.layer52(out51)
        out53 = torch.cat([out52, out46], 2)
        out54 = self.layer54(out53, out9)
        out55 = out54[0]
        out55 = out55 + out51
        out57 = self.layer57(out55)
        out58 = torch.cat([out57, out46], 2)
        out59 = self.layer59(out58, out8)
        out60 = out59[0]
        out60 = out60 + out55
        out62 = self.layer62(out60)
        out63 = torch.cat([out62, out46], 2)
        out64 = self.layer64(out63, out7)
        out65 = out64[0]
        out65 = out65 + out60
        out67 = self.layer67(out65)
        out68 = torch.cat([out67, out46], 2)
        out69 = self.layer69(out68, out6)
        out70 = out69[0]
        out70 = out70 + out65
        out72 = self.layer72(out70)
        out73 = torch.cat([out72, out46], 2)
        out74 = self.layer74(out73, out5)
        out75 = out74[0]
        out75 = out75 + out70
        out77 = self.layer77(out75)
        out78 = torch.cat([out77, out46], 2)
        out79 = self.layer79(out78, out4)
        out80 = out79[0]
        out80 = out80 + out75
        out82 = self.layer82(out80)
        return out82

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
