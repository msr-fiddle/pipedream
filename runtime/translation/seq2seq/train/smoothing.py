# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, padding_idx, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param padding_idx: index of the PAD token
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, output, target):
        x = output.view(output.size(0) * output.size(1), -1)
        logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)

        non_pad_mask = (target != self.padding_idx)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.sum()
        loss /= output.size(1)
        return loss


class CrossEntropyWrapper(nn.Module):
    def __init__(self, weight, size_average):
        super(CrossEntropyWrapper, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, size_average=size_average)

    def forward(self, output, target):
        x = output.view(output.size(0) * output.size(1), -1)
        loss = self.cross_entropy(x, target)
        loss /= output.size(1)
        return loss
