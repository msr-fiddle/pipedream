# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
import torch

import sys
sys.path.append("../..")
import communication

NUM_TRIALS = 20


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--master_addr", required=True, type=str,
                        help="IP address of master")
    parser.add_argument("--rank", required=True, type=int,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")
    parser.add_argument("--broadcast", action='store_true',
                        help="Broadcast within a server")

    args = parser.parse_args()

    num_ranks_in_server = 1
    if args.broadcast:
        num_ranks_in_server = 2
    local_rank = args.rank
    torch.cuda.set_device(local_rank)
    print("Local rank: %d" % local_rank)

    comm_handler = communication.CommunicationHandler(
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        local_rank=args.rank,
        num_ranks_in_server=num_ranks_in_server,
        world_size=2,
        fp16=False,
        backend='gloo'
    )

    tensor_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000,
                    100000000, 800000000]

    receive_ranks = {}
    send_ranks = {}
    tensor_tags = {}
    training_tensor_dtypes = {}
    tensor_shapes = {}
    for tag, tensor_size in enumerate(tensor_sizes):
        tensor_name = "out%d" % tag
        if args.rank == 1:
            receive_ranks[tensor_name] = [0]
        else:
            send_ranks[tensor_name] = [1]
        tensor_tags[tensor_name] = tag
        training_tensor_dtypes[tensor_name] = torch.float32
        tensor_shapes[tensor_name] = (tensor_size,)

    # Populate fields for ack.
    tensor_tags["ack"] = tag + 1
    tensor_shapes["ack"] = (1,)

    ranks_in_previous_stage = [] if args.rank == 0 else [0]
    ranks_in_next_stage = [1] if args.rank == 0 else []

    comm_handler.initialize(
        receive_ranks=receive_ranks,
        send_ranks=send_ranks,
        tensor_tags=tensor_tags,
        target_tensor_names=[],
        training_tensor_dtypes=training_tensor_dtypes,
        rank_in_stage=0,
        num_ranks_in_stage=1,
        ranks_in_previous_stage=ranks_in_previous_stage,
        ranks_in_next_stage=ranks_in_next_stage)
    comm_handler.set_tensor_shapes(tensor_shapes)
    comm_handler.start_helper_threads(num_iterations=NUM_TRIALS,
                                      forward_only=True)

    for i, tensor_size in enumerate(tensor_sizes):
        tensor_name = "out%d" % i
        if args.rank == 0:
            tensor = torch.tensor(range(tensor_size),
                                  dtype=torch.float32).cuda(local_rank)
            torch.distributed.barrier()
            start_time = time.time()
            for j in range(NUM_TRIALS):
                comm_handler.send(tensor_name, tensor,
                                  forward_minibatch_id=j,
                                  backward_minibatch_id=j,
                                  backward=False)
        else:
            torch.distributed.barrier()
            start_time = time.time()
            for j in range(NUM_TRIALS):
                tensor = comm_handler.recv(tensor_name,
                                           forward_minibatch_id=j,
                                           backward_minibatch_id=j,
                                           backward=False)
        torch.distributed.barrier()
        average_time = (time.time() - start_time) / NUM_TRIALS
        if args.rank == 1:  # Only time recvs since sends are asynchronous.
            print("Time to receive %s MB: %.3f seconds" % (
                (tensor_size * 4) / 10**6,
                average_time))
            throughput = (tensor_size * 4) / average_time
            print("Throughput: %.3f GB/s" % (throughput / 10**9))

    # Send and receive acks to flush the ack helper threads.
    ack_tensor = torch.zeros((1,), dtype=torch.int64).cuda()
    for j in range(NUM_TRIALS):
        if args.rank == 1:
             comm_handler.send("ack", ack_tensor,
                               forward_minibatch_id=j,
                               backward_minibatch_id=j,
                               backward=True)
        else:
            comm_handler.recv("ack",
                              forward_minibatch_id=j,
                              backward_minibatch_id=j,
                              backward=True)

    comm_handler.wait()
