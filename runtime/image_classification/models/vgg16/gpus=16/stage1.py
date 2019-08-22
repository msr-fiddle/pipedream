# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Dropout(p=0.5)
        self.layer4 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Dropout(p=0.5)
        self.layer7 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        return out7

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
