# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from .layers import CellStem0, CellStem1, FirstCell, NormalCell, ReductionCell

import torchmodules.torchgraph as torchgraph

class NASNet(nn.Module):
    def __init__(self, num_stem_features, num_normal_cells, filters, scaling, skip_reduction, use_aux=True,
                 num_classes=1000):
        super(NASNet, self).__init__()
        self.num_normal_cells = num_normal_cells
        self.skip_reduction = skip_reduction
        self.use_aux = use_aux
        self.num_classes = num_classes

        self.conv0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_stem_features, kernel_size=3, stride=2, bias=False)),
            ('bn', nn.BatchNorm2d(num_stem_features, eps=0.001, momentum=0.1, affine=True))
        ]))

        self.cell_stem_0 = CellStem0(in_channels=num_stem_features,
                                     out_channels=int(filters * scaling ** (-2)))
        self.cell_stem_1 = CellStem1(in_channels_x=int(4 * filters * scaling ** (-2)),
                                     in_channels_h=num_stem_features,
                                     out_channels=int(filters * scaling ** (-1)))

        x_channels = int(4 * filters * scaling ** (-1))
        h_channels = int(4 * filters * scaling ** (-2))
        cell_id = 0
        branch_out_channels = filters
        for i in range(3):
            self.add_module('cell_{:d}'.format(cell_id), FirstCell(
                in_channels_left=h_channels, out_channels_left=branch_out_channels // 2, in_channels_right=x_channels,
                out_channels_right=branch_out_channels))
            cell_id += 1
            h_channels = x_channels
            x_channels = 6 * branch_out_channels  # normal: concat 6 branches
            for _ in range(num_normal_cells - 1):
                self.add_module('cell_{:d}'.format(cell_id), NormalCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels, in_channels_right=x_channels,
                    out_channels_right=branch_out_channels))
                h_channels = x_channels
                cell_id += 1
            if i == 1 and self.use_aux:
                self.aux_features = nn.Sequential(
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3),
                                 padding=(2, 2), count_include_pad=False),
                    nn.Conv2d(in_channels=x_channels, out_channels=128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.1, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=768,
                              kernel_size=((14 + 2) // 3, (14 + 2) // 3), bias=False),
                    nn.BatchNorm2d(num_features=768, eps=1e-3, momentum=0.1, affine=True),
                    nn.ReLU()
                )
                self.aux_linear = nn.Linear(768, num_classes)
            # scaling
            branch_out_channels *= scaling
            if i < 2:
                self.add_module('reduction_cell_{:d}'.format(i), ReductionCell(
                    in_channels_left=h_channels, out_channels_left=branch_out_channels,
                    in_channels_right=x_channels, out_channels_right=branch_out_channels))
                x_channels = 4 * branch_out_channels  # reduce: concat 4 branches

        self.linear = nn.Linear(x_channels, self.num_classes)  # large: 4032; mobile: 1056

        self.num_params = sum([param.numel() for param in self.parameters()])
        if self.use_aux:
            self.num_params -= sum([param.numel() for param in self.aux_features.parameters()])
            self.num_params -= sum([param.numel() for param in self.aux_linear.parameters()])

    def features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        prev_x, x = x_stem_0, x_stem_1
        cell_id = 0
        for i in range(3):
            for _ in range(self.num_normal_cells):
                new_x = self._modules['cell_{:d}'.format(cell_id)](x, prev_x)
                prev_x, x = x, new_x
                cell_id += 1
            if i == 1 and self.training and self.use_aux:
                x_aux = self.aux_features(x)
            if i < 2:
                new_x = self._modules['reduction_cell_{:d}'.format(i)](x, prev_x)
                prev_x = x if not self.skip_reduction else prev_x
                x = new_x
        if self.training and self.use_aux:
            return [x, x_aux]
        return [x]

    def logits(self, features):
        relu = nn.ReLU(inplace=False)
        x = relu(features)
        kernel_size = x.size(2)
        if isinstance(kernel_size, torchgraph.TensorWrapper):
            kernel_size = kernel_size.tensor
        avgpool2d = nn.AvgPool2d(kernel_size=kernel_size)
        x = avgpool2d(x).view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        output = self.logits(x[0])
        if self.training and self.use_aux:
            x_aux = x[1].view(x[1].size(0), -1)
            aux_output = self.aux_linear(x_aux)
            return output, aux_output
        return output


def nasnetamobile(num_classes=1000):
    return NASNet(32, 4, 44, 2, skip_reduction=False, use_aux=True, num_classes=num_classes)


def nasnetalarge(num_classes=1000):
    return NASNet(96, 6, 168, 2, skip_reduction=True, use_aux=True, num_classes=num_classes)
