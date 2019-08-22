# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer3 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer10 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer19 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer20 = torch.nn.ReLU(inplace=True)
        self.layer21 = torch.nn.Dropout(p=0.5)
        self.layer22 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer23 = torch.nn.ReLU(inplace=True)
        self.layer24 = torch.nn.Dropout(p=0.5)
        self.layer25 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

        self._initialize_weights()

    def forward(self, input0):
        out0 = input0.clone()
        out3 = self.layer3(out0)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = out16.size(0)
        out18 = out16.view(out17, -1)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        return out25

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)