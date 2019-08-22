# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import threading
import time
import torch
import torch.distributed as dist

NUM_TRIALS = 20

def all_reduce_helper(tensor, group, multiplier, num_iterations):
    dist.barrier()
    start_time = time.time()
    for i in range(num_iterations):
        dist.all_reduce(tensor=tensor, group=group)
    dist.barrier()
    size = tensor.size()[0]
    bandwidth = (size * 4. * NUM_TRIALS * multiplier) / ((time.time() - start_time) * 10**6)
    print("Bandwidth for tensor size %s: %.2f MB/s" % (size, bandwidth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument("--master_addr", required=True, type=str,
                        help="IP address of master")
    parser.add_argument("--rank", required=True, type=int,
                        help="Rank of current worker")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local rank of current worker")
    parser.add_argument("--world_size", required=True, type=int,
                        help="World size")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group(args.backend, rank=args.rank, world_size=args.world_size)

    tensor_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

    groups = []
    for tag in range(len(tensor_sizes)):
        group = dist.new_group(list(range(args.world_size)))
        groups.append(group)

    multiplier = (2. * (args.world_size-1)) / args.world_size
    for tag, tensor_size in enumerate(tensor_sizes):
        group = groups[tag]
        tensor = torch.tensor(range(tensor_size), dtype=torch.float32).cuda(args.local_rank)
        all_reduce_helper(tensor, group, multiplier, NUM_TRIALS)
