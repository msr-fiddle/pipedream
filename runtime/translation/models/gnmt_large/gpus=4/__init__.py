# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .gnmt_large import GNMTSplit
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

def arch():
    return "gnmt_large"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1"], ["out0", "out1"]),
        (Stage1(), ["out0", "input1", "out1", "input2"], ["out12", "out13"]),
        (Stage2(), ["out12", "out13"], ["out14", "out15", "out12again"]),
        (Stage3(), ["out12again", "out14", "out15"], ["out18"]),
        (criterion, ["out18"], ["loss"])
    ]

def full_model():
    return GNMTSplit()
