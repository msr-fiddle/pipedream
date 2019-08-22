# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Dropout(p=0.5)
        self.layer4 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out6
