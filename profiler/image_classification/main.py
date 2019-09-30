# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys; sys.path = [".."] + sys.path
import torchmodules.torchgraph as torchgraph
import torchmodules.torchlogger as torchlogger
import torchmodules.torchprofiler as torchprofiler
import torchmodules.torchsummary as torchsummary

import argparse
from collections import OrderedDict
import os
import shutil
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
import torchvision.datasets as datasets
import torchvision.models as models

import models.densenet as densenet
import models.inception as inception
import models.mobilenet as mobilenet
import models.nasnet as nasnet
import models.resnext as resnext
import models.squeezenet as squeezenet

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mobilenet.__dict__[name]))
model_names += sorted(name for name in nasnet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(nasnet.__dict__[name]))
model_names += sorted(name for name in resnext.__dict__
    if name.islower() and not name.startswith("__")
    and callable(resnext.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_activations', action='store_true',
                    help="Log activations")
parser.add_argument('--log_activations_freq', default=5, type=int,
                    help="Frequency at which activations and gradients should be logged")
parser.add_argument('--log_activations_directory', default="activations",
                    help="Activations directory")
parser.add_argument('--profile_directory', default="profiles/",
                    help="Profile directory")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--log_reduce_times', action='store_true',
                    help="Log reduce times")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose', action='store_true',
                    help="Controls verbosity while profiling")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


def create_graph(model, train_loader, summary, directory):
    """Given a model, creates and visualizes the computation DAG
       of the model in the passed-in directory."""
    graph_creator = torchgraph.GraphCreator(model, summary, module_whitelist=[])
    graph_creator.hook_modules(model)
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if i >= 0:
            break
    graph_creator.unhook_modules()
    graph_creator.persist_graph(directory)

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
    args.profile = True

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('densenet'):
            model = densenet.__dict__[args.arch]()
        elif args.arch.startswith('inception_v3'):
            model = inception.__dict__[args.arch]()
        elif args.arch.startswith('mobilenet'):
            model = mobilenet.__dict__[args.arch]()
        elif args.arch.startswith('nasnet'):
            model = nasnet.__dict__[args.arch]()
        elif args.arch.startswith('resnext'):
            model = resnext.__dict__[args.arch]()
        elif args.arch.startswith('squeezenet'):
            model = squeezenet.__dict__[args.arch]()
        else:
            if args.arch not in models.__dict__:
                raise Exception("Invalid model architecture")
            model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        elif args.profile:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        if args.log_reduce_times:
            model = torch.nn.parallel.DistributedDataParallel(model, log_reduce_times=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch == 'inception_v3':
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
        )
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 299, 299), len(train_dataset))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if args.synthetic_data:
            train_dataset = SyntheticDataset((3, 224, 224), len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.profile:
        print("Collecting profile...")
        for i, (model_input, _) in enumerate(train_loader):
            model_input = model_input.cuda()
            if i >= 0:
                break
        summary = torchsummary.summary(model=model, module_whitelist=[], model_input=(model_input,),
                                       verbose=args.verbose, device="cuda")
        per_layer_times, data_time = profile_train(train_loader, model, criterion, optimizer)
        summary_i = 0
        per_layer_times_i = 0
        while summary_i < len(summary) and per_layer_times_i < len(per_layer_times):
            summary_elem = summary[summary_i]
            per_layer_time = per_layer_times[per_layer_times_i]
            if str(summary_elem['layer_name']) != str(per_layer_time[0]):
                summary_elem['forward_time'] = 0.0
                summary_elem['backward_time'] = 0.0
                summary_i += 1
                continue
            summary_elem['forward_time'] = per_layer_time[1]
            summary_elem['backward_time'] = per_layer_time[2]
            summary_i += 1
            per_layer_times_i += 1
        summary.append(OrderedDict())
        summary[-1]['layer_name'] = 'Input'
        summary[-1]['forward_time'] = data_time
        summary[-1]['backward_time'] = 0.0
        summary[-1]['nb_params'] = 0.0
        summary[-1]['output_shape'] = [args.batch_size] + list(model_input.size()[1:])
        create_graph(model, train_loader, summary,
                     os.path.join(args.profile_directory, args.arch))
        print("...done!")
        return

    i = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.num_minibatches is not None:
            if i > 0:
                break
        i += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(train_loader, model, criterion, args.num_minibatches)
        else:
            train(train_loader, model, criterion, optimizer, epoch,
                  args.num_minibatches)

            if args.num_minibatches is None:
                # evaluate on validation set
                prec1 = validate(val_loader, model, criterion)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)


def profile_train(train_loader, model, criterion, optimizer):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    NUM_STEPS_TO_PROFILE = 100  # profile 100 steps or minibatches

    # switch to train mode
    model.train()

    layer_timestamps = []
    data_times = []

    iteration_timestamps = []
    opt_step_timestamps = []
    data_timestamps = []
    
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_pid = os.getpid()
        data_time = time.time() - start_time
        data_time_meter.update(data_time)
        with torchprofiler.Profiling(model, module_whitelist=[]) as p:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            if isinstance(output, tuple):
                loss = sum((criterion(output_elem, target) for output_elem in output))
            else:
                loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer_step_start = time.time()
            optimizer.step()

            end_time = time.time()
            iteration_time = end_time - start_time
            batch_time_meter.update(iteration_time)

            if i >= NUM_STEPS_TO_PROFILE:
                break
        p_str = str(p)
        layer_timestamps.append(p.processed_times())
        data_times.append(data_time)

        if args.verbose:
            print('End-to-end time: {batch_time.val:.3f} s ({batch_time.avg:.3f} s)'.format(
                  batch_time=batch_time_meter))

        iteration_timestamps.append({"start": start_time * 1000 * 1000,
                                     "duration": iteration_time * 1000 * 1000})
        opt_step_timestamps.append({"start": optimizer_step_start * 1000 * 1000,
                                    "duration": (end_time - optimizer_step_start) * 1000 * 1000, "pid": os.getpid()})
        data_timestamps.append({"start":  start_time * 1000 * 1000,
                                "duration": data_time * 1000 * 1000, "pid": data_pid})
        
        start_time = time.time()

    layer_times = []
    tot_accounted_time = 0.0
    if args.verbose:
        print("\n==========================================================")
        print("Layer Type    Forward Time (ms)    Backward Time (ms)")
        print("==========================================================")

    for i in range(len(layer_timestamps[0])):
        layer_type = str(layer_timestamps[0][i][0])
        layer_forward_time_sum = 0.0
        layer_backward_time_sum = 0.0
        for j in range(len(layer_timestamps)):
            layer_forward_time_sum += (layer_timestamps[j][i][2] / 1000)
            layer_backward_time_sum += (layer_timestamps[j][i][5] / 1000)
        layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                    layer_backward_time_sum / len(layer_timestamps)))
        if args.verbose:
            print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
        tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

    print()
    print("Total accounted time: %.3f ms" % tot_accounted_time)
    return layer_times, (sum(data_times) * 1000.0) / len(data_times)


def train(train_loader, model, criterion, optimizer, epoch, num_minibatches=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start_time = time.time()
    logger = torchlogger.ActivationAndGradientLogger(args.log_activations_directory)
    for i, (input, target) in enumerate(train_loader):
        if args.log_activations and epoch % args.log_activations_freq == 0 and i == 0:
            logger.hook_modules(model, epoch)

        if num_minibatches is not None and i >= num_minibatches:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        if isinstance(output, tuple):
            loss = sum((criterion(output_elem, target) for output_elem in output))
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0], target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.log_activations and epoch % args.log_activations_freq == 0 and i == 0:
            logger.unhook_modules(model)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, model, criterion, num_minibatches=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    epoch_start_time = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if num_minibatches is not None and i >= num_minibatches:
                break
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if args.forward_only:
            print('Epoch 0: %.3f seconds' % (time.time() - epoch_start_time))
            print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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
