# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

class ResNet50Partitioned(torch.nn.Module):
    def __init__(self):
        super(ResNet50Partitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()

        self._initialize_weights()

    def forward(self, input0):
        (out0, out1) = self.stage0(input0)
        (out3, out2) = self.stage1(out0, out1)
        (out4, out5) = self.stage2(out3, out2)
        out6 = self.stage3(out4, out5)
        return out6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
