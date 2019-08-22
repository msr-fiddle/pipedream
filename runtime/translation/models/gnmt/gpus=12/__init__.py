# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

def arch():
    return "gnmt"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["input1", "out0"], ["out4"]),
        (Stage2(), ["out4"], ["out5"]),
        (Stage3(), ["out5"], ["out7"]),
        (Stage4(), ["out7"], ["out8", "out7again"]),
        (Stage5(), ["out7again", "out8", "input1", "input2"], ["out11", "out14"]),
        (Stage6(), ["out11", "out14"], ["out15", "out16", "out14again"]),
        (Stage7(), ["out14again", "out15", "out16"], ["out18", "out14again2"]),
        (Stage8(), ["out14again2", "out18"], ["out19", "out18again"]),
        (Stage9(), ["out18again", "out19"], ["out20"]),
        (criterion, ["out20"], ["loss"])
    ]
