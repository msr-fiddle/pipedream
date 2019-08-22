# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
from contextlib import contextmanager
import logging.config
import time
import ctypes

import numpy as np
import torch
import torch.distributed as dist
from mlperf_compliance import mlperf_log


def generate_seeds(rng, size):
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def gnmt_print(*args, **kwargs):
    barrier()
    if get_rank() == 0:
        kwargs['stack_offset'] = 2
        mlperf_log.gnmt_print(*args, **kwargs)


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


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


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()

@contextmanager
def timer(name):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    stop = time.time()
    elapsed = stop - start
    logging.info(f'TIMER {name} {elapsed}')

def setup_logging(log_file=os.devnull):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            self.rank = rank

        def filter(self, record):
            record.rank = self.rank
            return True

    rank = get_rank()
    rank_filter = RankFilter(rank)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(rank)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


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


def debug_tensor(tensor, name):
    """
    Simple utility which helps with debugging.
    Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    INFs.

    :param tensor: torch tensor
    :param name: name of the tensor (only for logging)
    """
    logging.info(name)
    tensor = tensor.detach().float().cpu().numpy()
    logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
                 f'AVG: {tensor.mean()} STD: {tensor.std()} '
                 f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')

def l2_promote():
    # Check what's the device limit for current device, should be 64 by default
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    result = torch.cuda.cudart().cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    result = torch.cuda.cudart().cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))

    # Get the device limit again, should be 128
    result = torch.cuda.cudart().cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    logging.info(f'L2 promotion: {pValue[0]}B')


lib = ctypes.cdll.LoadLibrary(None)
lib.THCudaHalfTensor_normall.argtypes=[ctypes.c_void_p, ctypes.c_void_p]
lib.THCudaHalfTensor_normall.restype = ctypes.c_float

def fused_norm(input):
    if input.type() == 'torch.cuda.HalfTensor':
        # 16384 is half 2 if you stare at it long enough
        return lib.THCudaHalfTensor_normall(torch.cuda._state_cdata,
            input._cdata, 16384)
    else:
        return(input.norm())

