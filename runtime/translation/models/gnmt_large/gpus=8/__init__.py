# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .gnmt_large import GNMT16Partitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5
from .stage6 import Stage6
from .stage7 import Stage7

def arch():
    return "gnmt_large"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1"], ["out1", "out0"]),
        (Stage1(), ["out1", "out0"], ["out4", "out5"]),
        (Stage2(), ["input1", "out4", "out5", "input2"], ["out13", "out14"]),
        (Stage3(), ["out13", "out14"], ["out13_1", "out15", "out16"]),
        (Stage4(), ["out13_1", "out15", "out16"], ["out13_2", "out18", "out19"]),
        (Stage5(), ["out13_2", "out18", "out19"], ["out13_3", "out20", "out21"]),
        (Stage6(), ["out13_3", "out20", "out21"], ["out23"]),
        (Stage7(), ["out23"], ["out24"]),
        (criterion, ["out24"], ["loss"])
    ]

def full_model():
    return GNMT16Partitioned()
