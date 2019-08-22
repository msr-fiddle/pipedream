# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from torch.utils.data.sampler import Sampler

from seq2seq.utils import get_world_size, get_rank


class BucketingSampler(Sampler):
    """
    Distributed data sampler supporting bucketing by sequence length.
    """
    def __init__(self, dataset, batch_size, seeds, bucketing=True,
                 world_size=None, rank=None, sort=False):
        """
        Constructor for the BucketingSampler.

        :param dataset: dataset
        :param batch_size: batch size
        :param bucketing: if True enables bucketing by sequence length
        :param world_size: number of processes participating in distributed
            training
        :param rank: rank of the current process within world_size
        """
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.bucketing = bucketing
        self.seeds = seeds
        self.sort = sort

        self.batch_size = batch_size
        self.global_batch_size = batch_size * world_size

        self.data_len = len(self.dataset)
        self.num_samples = self.data_len // self.global_batch_size \
            * self.global_batch_size

    def __iter__(self):
        # gnmt_print(key=mlperf_log.INPUT_ORDER)

        # deterministically shuffle based on epoch
        g = torch.Generator()
        seed = self.seeds[self.epoch]
        # logging.info(f'Sampler for epoch {self.epoch} uses seed {seed}')
        g.manual_seed(seed)

        if self.sort:
            indices = range(self.num_samples)
        else:
            # generate permutation
            indices = torch.randperm(self.data_len, generator=g)
            # make indices evenly divisible by (batch_size * world_size)
            indices = indices[:self.num_samples]

        if self.sort:
            print ("Sorting inputs from smallest to largest")
            lengths = self.dataset.lengths[:self.num_samples]
            info = zip(lengths, indices)

            def get_length(item):
                return item[0]

            sorted_info = sorted(info, key=get_length)
            indices = torch.tensor([x[1] for x in sorted_info])


        # splits the dataset into chunks of 'batches_in_shard' global batches
        # each, sorts by (src + tgt) sequence length within each chunk,
        # reshuffles all global batches
        if self.bucketing and not self.sort:
            batches_in_shard = 80
            shard_size = self.global_batch_size * batches_in_shard
            # gnmt_print(key=mlperf_log.INPUT_SHARD, value=shard_size)
            nshards = (self.num_samples + shard_size - 1) // shard_size

            lengths = self.dataset.lengths[indices]

            shards = [indices[i * shard_size:(i+1) * shard_size] for i in range(nshards)]
            len_shards = [lengths[i * shard_size:(i+1) * shard_size] for i in range(nshards)]

            indices = []
            for len_shard in len_shards:
                _, ind = len_shard.sort()
                indices.append(ind)

            output = tuple(shard[idx] for shard, idx in zip(shards, indices))
            indices = torch.cat(output)

            # global reshuffle
            indices = indices.view(-1, self.global_batch_size)
            order = torch.randperm(indices.shape[0], generator=g)
            indices = indices[order, :]
            indices = indices.view(-1)

        assert len(indices) == self.num_samples

        # build indices for each individual worker
        # consecutive ranks are getting consecutive batches,
        # default pytorch DistributedSampler assigns strided batches
        # with offset = length / world_size
        indices = indices.view(-1, self.batch_size)
        indices = indices[self.rank::self.world_size].contiguous()
        indices = indices.view(-1)
        indices = indices.tolist()

        assert len(indices) == self.num_samples // self.world_size

        return iter(indices)

    def __len__(self):
        return self.num_samples // self.world_size

    def set_epoch(self, epoch):
        """
        Sets current epoch index. This value is used to seed RNGs in __iter__()
        function.

        :param epoch: index of current epoch
        """
        self.epoch = epoch


class StaticDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, pad, world_size=None, rank=None):
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()

        self.world_size = world_size

        global_batch_size = batch_size * world_size

        data_len = len(dataset)
        num_samples = (data_len + global_batch_size - 1) \
            // global_batch_size * global_batch_size
        self.num_samples = num_samples

        indices = list(range(data_len))
        if pad:
            indices += [0] * (num_samples - len(indices))
        else:
            indices += [-1] * (num_samples - len(indices))
        indices = torch.tensor(indices)

        indices = indices.view(-1, batch_size)
        indices = indices[rank::world_size].contiguous()
        indices = indices.view(-1)
        indices = indices[indices != -1]
        indices = indices.tolist()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
