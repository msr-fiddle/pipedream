# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from seq2seq.models.decoder import Classifier

class Stage7(torch.nn.Module):
    def __init__(self):
        super(Stage7, self).__init__()
        self.layer1 = Classifier(1024, 32320)

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        return out1
