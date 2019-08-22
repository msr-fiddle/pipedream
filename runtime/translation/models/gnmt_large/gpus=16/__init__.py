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
from .stage10 import Stage10
from .stage11 import Stage11
from .stage12 import Stage12
from .stage13 import Stage13

def arch():
    return "gnmt_large"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1"], [ "out0"]),
        (Stage1(), ["out0"], ["out4", "out3"]),
        (Stage2(), ["out4", "out3"], ["out5", "out6"]),
        (Stage3(), ["out5", "out6"], ["out7", "out8"]),
        (Stage4(), ["out7", "out8"], ["out9"]),
        (Stage5(), ["input1", "input2", "out9"], ["out10", "out12"]),
        (Stage6(), ["out10", "out12"], ["out18", "out19", "out20"]),
        (Stage7(), ["out18", "out19", "out20"], ["out21", "out22", "out18again"]),
        (Stage8(), ["out18again", "out21", "out22"], ["out23", "out24", "out18again2"]),
        (Stage9(), ["out18again2", "out23", "out24"], ["out25", "out26", "out18again3"]),
        (Stage10(), ["out25", "out26"], ["out27", "out28"]),
        (Stage11(), ["out18again3", "out27", "out28"], ["out30", "out31"]),
        (Stage12(), ["out31"], ["out32"]),
        (Stage13(), ["out30", "out32"], ["out33"]),
        (criterion, ["out33"], ["loss"])
    ]
