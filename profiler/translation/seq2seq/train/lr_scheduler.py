# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import math


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        lr_method="mlperf",
        warmup_iters=100,
        remain_steps=600,
        decay_steps=100,
        last_epoch=-1,
    ):
        if lr_method not in ("none", "mlperf"):
            raise ValueError(
                "Only 'none' or 'mlperf' warmup_method accepted"
                "got {}".format(lr_method)
            )
        self.lr_method = lr_method
        self.warmup_iters = warmup_iters # iterations before it reaches base LR
        self.remain_steps = remain_steps # iteration at which decay starts
        self.decay_steps = decay_steps # number of steps between each decay

        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.lr_method == "none":
            return [base_lr for base_lr in self.base_lrs]
        elif self.last_epoch <= self.warmup_iters:
            # MLPerf warmup Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
            warmup_factor = math.exp(math.log(0.01) / self.warmup_iters)
            inv_decay = warmup_factor ** (self.warmup_iters - self.last_epoch)
            return [base_lr * inv_decay for base_lr in self.base_lrs]
        elif self.last_epoch >= self.remain_steps:
            num_decay_steps = min(int((self.last_epoch - self.remain_steps) / self.decay_steps) + 1, 4)
            return [
                base_lr * (0.5 ** num_decay_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]

