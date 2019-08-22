# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5
from .stage6 import Stage6
from .stage7 import Stage7
from .stage8 import Stage8
from .stage9 import Stage9
from .stage10 import Stage10
from .stage11 import Stage11
from .stage12 import Stage12
from .stage13 import Stage13
from .stage14 import Stage14
from .stage15 import Stage15

class VGG16Split(torch.nn.Module):
    def __init__(self):
        super(VGG16Split, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()
        self.stage5 = Stage5()
        self.stage6 = Stage6()
        self.stage7 = Stage7()
        self.stage8 = Stage8()
        self.stage9 = Stage9()
        self.stage10 = Stage10()
        self.stage11 = Stage11()
        self.stage12 = Stage12()
        self.stage13 = Stage13()
        self.stage14 = Stage14()
        self.stage15 = Stage15()
        self._initialize_weights()

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        out5 = self.stage5(out4)
        out6 = self.stage6(out5)
        out7 = self.stage7(out6)
        out8 = self.stage8(out7)
        out9 = self.stage9(out8)
        out10 = self.stage10(out9)
        out11 = self.stage11(out10)
        out12 = self.stage12(out11)
        out13 = self.stage13(out12)
        out14 = self.stage14(out13)
        out15 = self.stage15(out14)
        return out15

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
