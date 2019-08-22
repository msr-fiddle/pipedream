# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import Classifier

class Stage9(torch.nn.Module):
    def __init__(self):
        super(Stage9, self).__init__()
        self.layer4 = Classifier(1024, 32320)

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = out0
        out2 = out2 + out1
        out4 = self.layer4(out2)
        return out4

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
