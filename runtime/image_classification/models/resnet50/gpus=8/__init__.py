# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .resnet50 import ResNet50Split
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

def arch():
    return "resnet50"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out3", "out2"]),
        (Stage2(), ["out3", "out2"], ["out4"]),
        (criterion, ["out4"], ["loss"])
    ]

def full_model():
    return ResNet50Split()
