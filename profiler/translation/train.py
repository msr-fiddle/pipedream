# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python

import sys; sys.path = [".."] + sys.path

import argparse
import os
import logging
from ast import literal_eval
import random

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.optim
from mlperf_compliance import mlperf_log

from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.data.dataset import TextDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.utils import setup_logging
from seq2seq.utils import barrier
from seq2seq.utils import get_rank
from seq2seq.utils import gnmt_print
from seq2seq.utils import get_world_size
from seq2seq.utils import l2_promote
from seq2seq.utils import broadcast_seeds
from seq2seq.utils import generate_seeds
import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
from seq2seq.inference.inference import Translator


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default=None, required=True,
                         help='path to directory with training/validation \
                         data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                         dataset (useful for benchmarking), by default \
                         uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='results',
                         help='path to directory with results, it it will be \
                         automatically created if does not exist')
    results.add_argument('--save', default='gnmt_wmt16',
                         help='defines subdirectory within RESULTS_DIR for \
                         results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--model-config',
                       default="{'hidden_size': 1024,'num_layers': 4, "
                       "'dropout': 0.2, 'share_embedding': True}",
                       help='GNMT architecture configuration')
    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='set random number generator seed')
    general.add_argument('--disable-eval', action='store_true', default=False,
                         help='disables validation after every epoch')

    cuda = general.add_mutually_exclusive_group(required=False)
    cuda.add_argument('--cuda', dest='cuda', action='store_true',
                      help='enables cuda (use \'--no-cuda\' to disable)')
    cuda.add_argument('--no-cuda', dest='cuda', action='store_false',
                      help=argparse.SUPPRESS)
    cuda.set_defaults(cuda=True)

    cudnn = general.add_mutually_exclusive_group(required=False)
    cudnn.add_argument('--cudnn', dest='cudnn', action='store_true',
                       help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                       help=argparse.SUPPRESS)
    cudnn.set_defaults(cudnn=True)

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--batch-size', default=128, type=int,
                          help='batch size for training')
    training.add_argument('--epochs', default=8, type=int,
                          help='number of total epochs to run')
    training.add_argument('--optimization-config', type=str,
                          default="{'optimizer': 'Adam', 'lr': 5e-4, "
                          "'betas':(0.9,0.999)}",
                          help='optimizer config')
    training.add_argument('--scheduler-config', type=str,
                          default="{'lr_method':'none', 'warmup_iters':0, "
                          "'remain_steps':0, 'decay_steps':0}",
                          help='scheduler config')
    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enabled gradient clipping and sets maximum \
                          gradient norm value')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training')
    training.add_argument('--train-loader-workers', default=2, type=int,
                          help='number of workers for training data loading')
    training.add_argument('--arch', required=True, type=str,
                          help='Architecture name')

    bucketing = training.add_mutually_exclusive_group(required=False)
    bucketing.add_argument('--bucketing', dest='bucketing',
                           action='store_true',
                           help='enables bucketing (use \'--no-bucketing\' to \
                           disable)')
    bucketing.add_argument('--no-bucketing', dest='bucketing',
                           action='store_false', help=argparse.SUPPRESS)
    bucketing.set_defaults(bucketing=True)

    # validation
    validation = parser.add_argument_group('validation setup')
    validation.add_argument('--val-batch-size', default=64, type=int,
                            help='batch size for validation')
    validation.add_argument('--max-length-val', default=150, type=int,
                            help='maximum sequence length for validation')
    validation.add_argument('--min-length-val', default=0, type=int,
                            help='minimum sequence length for validation')
    validation.add_argument('--val-loader-workers', default=0, type=int,
                            help='number of workers for validation data \
                            loading')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--max-length-test', default=150, type=int,
                      help='maximum sequence length for test')
    test.add_argument('--min-length-test', default=0, type=int,
                      help='minimum sequence length for test')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--target-bleu', default=None, type=float,
                      help='target accuracy')
    test.add_argument('--intra-epoch-eval', default=0, type=int,
                      help='evaluate within epoch')
    test.add_argument('--test-loader-workers', default=0, type=int,
                      help='number of workers for test data loading')

    # checkpointing
    checkpoint = parser.add_argument_group('checkpointing setup')
    checkpoint.add_argument('--start-epoch', default=0, type=int,
                            help='manually set initial epoch counter')
    checkpoint.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='resumes training from checkpoint from PATH')
    checkpoint.add_argument('--save-all', action='store_true', default=False,
                            help='saves checkpoint after every epoch')
    checkpoint.add_argument('--save-freq', default=5000, type=int,
                            help='save checkpoint every SAVE_FREQ batches')
    checkpoint.add_argument('--keep-checkpoints', default=0, type=int,
                            help='keep only last KEEP_CHECKPOINTS checkpoints, \
                            affects only checkpoints controlled by \
                            --save-freq option')

    # distributed support
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')
    distributed.add_argument('--enable-apex-allreduce-overlap',
                             action='store_true', default=False,
                             help='enable overlap of allreduce communication \
                             with bprop')
    distributed.add_argument('--apex-message-size', default=1e7, type=int,
                             help='min. number of elements in communication \
                             bucket')

    return parser.parse_args()


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                   value='Cross Entropy')
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                   value='Cross Entropy with label smoothing')
        gnmt_print(key=mlperf_log.MODEL_HP_LOSS_SMOOTHING,
                   value=smoothing)

    return criterion


def main():
    """
    Launches data-parallel multi-gpu training.
    """
    mlperf_log.ROOT_DIR_GNMT = os.path.dirname(os.path.abspath(__file__))
    mlperf_log.LOGGER.propagate = False

    args = parse_args()

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialize distributed backend
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        assert args.cuda
        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        assert torch.distributed.is_initialized()

    gnmt_print(key=mlperf_log.RUN_START)

    args.rank = get_rank()

    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_gpu_{args.rank}.log'
    setup_logging(os.path.join(save_path, log_filename))

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    # setup L2 promotion
    if args.cuda:
        l2_promote()

    gnmt_print(key=mlperf_log.RUN_SET_RANDOM_SEED)
    # https://github.com/mlperf/policies/issues/120#issuecomment-431111348
    if args.seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logging.info(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        master_seed = args.seed
        logging.info(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, args.epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)

    # set worker seed
    worker_seed = worker_seeds[args.rank]
    logging.info(f'Worker {args.rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    # build tokenizer
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME))

    # build datasets
    gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    gnmt_print(key=mlperf_log.TRAIN_HP_MAX_SEQ_LEN,
               value=args.max_length_train)

    train_data = LazyParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size)

    gnmt_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
               value=len(train_data))

    val_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_VAL_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_VAL_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_val,
        max_len=args.max_length_val,
        sort=True)

    gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

    test_data = TextDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_test,
        max_len=args.max_length_test,
        sort=False)

    gnmt_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
               value=len(test_data))

    vocab_size = tokenizer.vocab_size
    # size of the vocabulary has been padded to a multiple of 8
    gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE,
               value=vocab_size)

    # build GNMT model
    model_config = dict(vocab_size=vocab_size, math=args.math,
                        **literal_eval(args.model_config))
    model = GNMT(**model_config)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
    opt_config = literal_eval(args.optimization_config)
    scheduler_config = literal_eval(args.scheduler_config)
    logging.info(f'Training optimizer: {opt_config}')
    logging.info(f'Training LR Schedule: {scheduler_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         seeds=shuffling_seeds,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         bucketing=args.bucketing,
                                         num_workers=args.train_loader_workers)

    gnmt_print(key=mlperf_log.INPUT_BATCH_SIZE,
               value=args.batch_size * get_world_size())
    gnmt_print(key=mlperf_log.INPUT_SIZE,
               value=train_loader.sampler.num_samples)

    val_loader = val_data.get_loader(batch_size=args.val_batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=args.val_loader_workers)

    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       pad=True,
                                       num_workers=args.test_loader_workers)

    gnmt_print(key=mlperf_log.EVAL_SIZE,
               value=len(test_loader.dataset))

    translator = Translator(model=model,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_length_test,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda,
                            print_freq=args.print_freq,
                            dataset_dir=args.dataset_dir,
                            target_bleu=args.target_bleu,
                            save_path=args.save_path)

    # create trainer
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info={'config': args, 'tokenizer': tokenizer.get_state()},
        opt_config=opt_config,
        scheduler_config=scheduler_config,
        batch_first=batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        print_freq=args.print_freq,
        cuda=args.cuda,
        distributed=distributed,
        distributed_overlap_allreduce=args.enable_apex_allreduce_overlap,
        distributed_overlap_allreduce_messagesize=args.apex_message_size,
        intra_epoch_eval=args.intra_epoch_eval,
        translator=translator,
        arch=args.arch)

    trainer_options['model'] = model
    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error(f'No checkpoint found at {args.resume}')

    # training loop
    # best_loss = float('inf')
    gnmt_print(key=mlperf_log.TRAIN_LOOP)

    for epoch in range(1):
        logging.info(f'Starting epoch {epoch}')
        gnmt_print(key=mlperf_log.TRAIN_EPOCH,
                   value=epoch)

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss, train_perf = trainer.optimize(train_loader)

        logging.info(f'Finished epoch {epoch}')

    # Save the checkpoint at the end of the training loop, after the RUN_STOP
    # tag
    # https://github.com/mlperf/policies/issues/55#issuecomment-428335773
    if not args.disable_eval:
        gnmt_print(key=mlperf_log.TRAIN_CHECKPOINT)
        if get_rank() == 0:
            trainer.save(save_all=args.save_all, is_best=True)

    gnmt_print(key=mlperf_log.RUN_FINAL)


if __name__ == '__main__':
    main()
