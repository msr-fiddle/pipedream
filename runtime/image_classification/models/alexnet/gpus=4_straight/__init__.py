# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .alexnet import AlexNetStraightSplit
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "alexnet"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"]),
        (Stage2(), ["out1"], ["out2"]),
        (Stage3(), ["out2"], ["out3"]),
        (criterion, ["out3"], ["loss"])
    ]

def full_model():
    return AlexNetStraightSplit()
