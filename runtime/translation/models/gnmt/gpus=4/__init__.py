# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .gnmt import GNMTSplit 
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "gnmt"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1"], ["out2", "out1"]),
        (Stage1(), ["out2", "input1", "input2", "out1"], ["out3", "out7"]),
        (Stage2(), ["out3", "out7"], ["out8", "out9", "out10"]),
        (Stage3(), ["out8", "out9", "out10"], ["out12"]),
        (criterion, ["out12"], ["loss"])
    ]

def full_model():
    return GNMTSplit()
