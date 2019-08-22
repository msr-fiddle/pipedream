# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.append("..")
import runtime
import sgd
import adam

from seq2seq.utils import l2_promote
from seq2seq.utils import build_gnmt_criterion
from seq2seq.data.tokenizer import Tokenizer
import seq2seq.data.config as config
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import ParallelDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--grad-clip', default=5.0, type=float,
                    help='enabled gradient clipping and sets maximum gradient norm value')
parser.add_argument('--eval-batch-size', default=100, type=int,
                    help='eval mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")

parser.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training')
parser.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training')
parser.add_argument('--no-bucketing', action='store_true',
                    help='enables bucketing')

# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

best_prec1 = 0


# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

# Synthetic Dataset class.
class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Special case handling for GNMT model
    l2_promote()

    torch.cuda.set_device(args.local_rank)

    # build tokenizer
    tokenizer = Tokenizer(os.path.join(args.data_dir, config.VOCAB_FNAME))

    # define loss function
    criterion = build_gnmt_criterion(
        vocab_size=tokenizer.vocab_size, padding_idx=config.PAD, smoothing=0.1)

    # create stages of the model
    module = importlib.import_module(args.module)
    args.arch = module.arch()
    model = module.model(criterion)

    input_size = [args.max_length_train, args.batch_size]
    training_tensor_shapes = {"input0": input_size, "input1": [args.batch_size],
                              "input2": input_size, "target": [args.max_length_train * args.batch_size],
                              "target_length": [args.batch_size]}
    dtypes = {"input0": torch.int64, "input1": torch.int64, "input2": torch.int64,
              "target": torch.int64, "target_length": torch.int32}
    inputs_module_destinations = {"input0": 0, "input1": 0, "input2": 0}
    target_tensor_names = {"target", "target_length"}
    for module_id, (stage, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage.cuda()
        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr,
        rank=args.rank, local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.TRANSLATION,
        enable_recompute=args.recompute)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # define optimizer
    if args.no_input_pipelining:
        num_versions = 1
    else:
        # number of versions is the total number of machines following the current
        # stage, shared amongst all replicas in this stage
        num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    # TODO: make this configurable by args
    use_adam_optimizer = True
    if use_adam_optimizer:
        optimizer = adam.AdamWithWeightStashing(
            modules=r.modules(), master_parameters=r.master_parameters,
            model_parameters=r.model_parameters, loss_scale=args.loss_scale,
            num_versions=num_versions, lr=args.lr, betas=(0.9,0.999),
            weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency,
            macrobatch=args.macrobatch)
    else:
        optimizer = sgd.SGDWithWeightStashing(
            modules=r.modules(), master_parameters=r.master_parameters,
            model_parameters=r.model_parameters, loss_scale=args.loss_scale,
            num_versions=num_versions, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_dataset = LazyParallelDataset(
        src_fname=os.path.join(args.data_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.data_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=None)

    val_dataset = ParallelDataset(
        src_fname=os.path.join(args.data_dir, config.SRC_VAL_FNAME),
        tgt_fname=os.path.join(args.data_dir, config.TGT_VAL_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=True)

    distributed_sampler = False
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            distributed_sampler = True

    # TODO: fix random seeds
    train_loader = train_dataset.get_loader(
        batch_size=args.batch_size, seeds=range(args.epochs),
        batch_first=False, shuffle=True,
        bucketing=not args.no_bucketing, num_workers=args.workers,
        world_size=r.num_ranks_in_first_stage,
        rank=r.rank_in_stage if r.stage == 0 else 0
    )

    val_loader = val_dataset.get_loader(
        batch_size=args.batch_size, batch_first=False,
        shuffle=True, num_workers=args.workers,
        world_size=r.num_ranks_in_first_stage,
        seeds=range(args.epochs),
        rank=r.rank_in_stage if r.stage == 0 else 0
    )

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)

    for epoch in range(args.start_epoch, args.epochs):
        if distributed_sampler:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.epochs, r, args.lr_policy)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train(train_loader, r, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, r, epoch)
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    'tokenizer': tokenizer.get_state()
                }, args.checkpoint_dir, r.stage, epoch)


def train(train_loader, r, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.train(n)
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    # start num_warmup_minibatches forward passes
    for i in range(num_warmup_minibatches):
        r.run_forward()

    for i in range(n - num_warmup_minibatches):
        # perform forward pass
        r.run_forward()

        if is_last_stage():
            # measure accuracy and record loss
            output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()
            losses.update(loss, num_tokens)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            epoch_time = (end - epoch_start_time) / 3600.0
            full_epoch_time = (epoch_time / float(i+1)) * float(n)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                      'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, n, batch_time=batch_time,
                       epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                       loss=losses, # top1=top1, top5=top5,
                       memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()
        else:
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                       epoch, i, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                       cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                import sys; sys.stdout.flush()

        # perform backward pass
        if args.fp16:
            r.zero_grad()
        else:
            optimizer.zero_grad()
        optimizer.load_old_params()

        r.run_backward()
        optimizer.load_new_params()
        optimizer.step()

    # finish remaining backward passes
    for i in range(num_warmup_minibatches):
        optimizer.zero_grad()
        optimizer.load_old_params()
        r.run_backward()
        optimizer.load_new_params()
        optimizer.step()

    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running validation for %d minibatches" % n)

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()

                # measure accuracy and record loss
                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, output.size(0))
                # top1.update(prec1[0], output.size(0))
                # top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


# TODO: Verify that checkpointing works correctly for GNMT
def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, total_epochs, r, lr_policy):
    """ Adjusts learning rate based on stage, epoch, and policy.

    Gets learning rate for stage from runtime and adjusts based on policy.

    Supported LR policies:
         - step
         - polynomial decay
         - exponential decay
    """
    stage_base_lr = r.get_adjusted_learning_rate(base_lr=args.lr)
    if lr_policy == "step":
        lr = stage_base_lr * (0.1 ** (epoch // 30))
    elif lr_policy == "polynomial":
        power = 2.0
        lr = stage_base_lr * ((1.0 - (float(epoch) / float(total_epochs))) ** power)
    elif lr_policy == "exponential_decay":
        decay_rate = 0.97
        lr = stage_base_lr * (decay_rate ** (float(epoch) / float(total_epochs)))
    else:
        raise NotImplementedError
    print("Epoch: %d\tLearning rate: %f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
