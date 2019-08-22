# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import torch


from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.train.smoothing import CrossEntropyWrapper


def l2_promote():
    # Check what's the device limit for current device, should be 64 by default
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    result = torch.cuda.cudart().cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    result = torch.cuda.cudart().cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))

    # Get the device limit again, should be 128
    result = torch.cuda.cudart().cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))


def build_gnmt_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = CrossEntropyWrapper(weight=loss_weight, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, skip_first=True):
        self.reset()
        self.skip = skip_first

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if self.skip:
            self.skip = False
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def reduce(self, op):
        """
        Reduces average value over all workers.

        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        if distributed:
            if(hasattr(dist, "get_backend")):
                backend = dist.get_backend()
            else:
                backend = dist._backend

            cuda = (backend == dist.dist_backend.NCCL)

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])
            dist.all_reduce(avg, op=dist.reduce_op.SUM)
            dist.all_reduce(_sum, op=dist.reduce_op.SUM)
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()
