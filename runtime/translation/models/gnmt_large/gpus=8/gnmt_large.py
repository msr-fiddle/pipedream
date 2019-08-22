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

class GNMT16Partitioned(torch.nn.Module):
    def __init__(self):
        super(GNMT16Partitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()
        self.stage5 = Stage5()
        self.stage6 = Stage6()
        self.stage7 = Stage7()

    def forward(self, input0, input1, input2):
        (out1, out0) = self.stage0(input0, input1)
        (out4, out5) = self.stage1(out1, out0)
        (out13, out14) = self.stage2(input1, out4, out5, input2)
        (out13_1, out15, out16) = self.stage3(out13, out14)
        (out13_2, out18, out19) = self.stage4(out13_1, out15, out16)
        (out13_3, out20, out21) = self.stage5(out13_2, out18, out19)
        out23 = self.stage6(out13_3, out20, out21)
        out24 = self.stage7(out23)
        return out24
