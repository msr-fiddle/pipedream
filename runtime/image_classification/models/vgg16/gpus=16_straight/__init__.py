# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vgg16 import VGG16Split 
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

def arch():
    return "vgg16"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"]),
        (Stage2(), ["out1"], ["out2"]),
        (Stage3(), ["out2"], ["out3"]),
        (Stage4(), ["out3"], ["out4"]),
        (Stage5(), ["out4"], ["out5"]),
        (Stage6(), ["out5"], ["out6"]),
        (Stage7(), ["out6"], ["out7"]),
        (Stage8(), ["out7"], ["out8"]),
        (Stage9(), ["out8"], ["out9"]),
        (Stage10(), ["out9"], ["out10"]),
        (Stage11(), ["out10"], ["out11"]),
        (Stage12(), ["out11"], ["out12"]),
        (Stage13(), ["out12"], ["out13"]),
        (Stage14(), ["out13"], ["out14"]),
        (Stage15(), ["out14"], ["out15"]),
        (criterion, ["out15"], ["loss"])
    ]

def full_model():
    return VGG16Split()
