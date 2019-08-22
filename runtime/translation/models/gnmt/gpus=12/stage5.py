# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import RecurrentAttention

class Stage5(torch.nn.Module):
    def __init__(self):
        super(Stage5, self).__init__()
        self.layer8 = torch.nn.Embedding(32320, 1024, padding_idx=0)
        self.layer10 = RecurrentAttention(1024, 1024, 1024)

    def forward(self, input1, input0, input2, input3):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = input2.clone()
        out3 = input3.clone()
        out5 = None
        out6 = None
        out7 = None
        out8 = self.layer8(out3)
        out0 = out0 + out1
        out10 = self.layer10(out8, out7, out0, out2)
        out11 = out10[2]
        out12 = out10[0]
        return (out12, out11)

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
