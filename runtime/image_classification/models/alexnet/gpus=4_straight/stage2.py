# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer1 = torch.nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer6 = torch.nn.ReLU(inplace=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out6
