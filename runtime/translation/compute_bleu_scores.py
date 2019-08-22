# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
import logging
import argparse
import importlib
import warnings
from ast import literal_eval

import torch
import torch.distributed as dist

from collections import OrderedDict
import os
import sys
sys.path.append("..")

from seq2seq.models.gnmt import GNMT
from seq2seq.inference.inference import Translator
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer


def parse_args():
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(description='GNMT Translate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en/',
                         help='path to directory with training/validation data')
    dataset.add_argument('-i', '--input', required=True,
                         help='full path to the input file (tokenized)')
    dataset.add_argument('-o', '--output', required=True,
                         help='full path to the output file (tokenized)')
    dataset.add_argument('-r', '--reference', default=None,
                         help='full path to the reference file (for sacrebleu)')
    dataset.add_argument('--checkpoint_path', required=True,
                         help='full path to the model checkpoint file')
    # parameters
    params = parser.add_argument_group('inference setup')
    params.add_argument('--num_layers', required=True, type=int,
                        help='number of layers in GNMT model')
    params.add_argument('--num_stages', required=True, type=int,
                        help='number of stages in split GNMT model')
    params.add_argument('--batch-size', default=128, type=int,
                        help='batch size per GPU')
    params.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    params.add_argument('--max-seq-len', default=80, type=int,
                        help='maximum generated sequence length')
    params.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    params.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    params.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')
    # general setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')

    bleu_parser = general.add_mutually_exclusive_group(required=False)
    bleu_parser.add_argument('--bleu', dest='bleu', action='store_true',
                             help='compares with reference and computes BLEU \
                             (use \'--no-bleu\' to disable)')
    bleu_parser.add_argument('--no-bleu', dest='bleu', action='store_false',
                             help=argparse.SUPPRESS)
    bleu_parser.set_defaults(bleu=True)

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

    cuda_parser = general.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true',
                             help='enables cuda (use \'--no-cuda\' to disable)')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                             help=argparse.SUPPRESS)
    cuda_parser.set_defaults(cuda=True)

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    general.add_argument('--print-freq', '-p', default=1, type=int,
                         help='print log every PRINT_FREQ batches')
    general.add_argument('--module', required=True,
                         help="Module to load")

    args = parser.parse_args()

    if args.bleu and args.reference is None:
        parser.error('--bleu requires --reference')

    return args


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.

    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.

    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def get_submodule_and_parameter_name(module, parameter_name):
    parameter_name_split = parameter_name.split(".")
    submodule = module
    for attribute_name in parameter_name_split[:-1]:
        submodule = getattr(submodule, attribute_name)
    return (submodule, parameter_name_split[-1])


def main():
    """
    Launches translation (inference).
    Inference is executed on a single GPU, implementation supports beam search
    with length normalization and coverage penalty.
    """
    args = parse_args()
    args.batch_first = False

    if args.cuda:
        torch.cuda.set_device(0)
    if not args.cuda and torch.cuda.is_available():
        warnings.warn('cuda is available but not enabled')
    if args.math == 'fp16' and not args.cuda:
        raise RuntimeError('fp16 requires cuda')
    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    num_stages = args.num_stages
    # compute BLEU score for every epoch
    print("Epoch\tBLEU score")
    epoch = 0
    while True:
        # no more epochs to run, since desired file not available
        if not os.path.isfile(os.path.join(args.checkpoint_path,
                                           f"checkpoint.0.pth.tar.epoch.{epoch}")):
            break

        module = importlib.import_module(args.module)
        model = module.model(None)
        num_modules = len(model)

        key_to_module_mapping = OrderedDict()
        all_stages_state_dict = OrderedDict()
        module_id = 0
        stage_id = 0
        for stage_id in range(num_stages):
            # load the checkpoint associated with a stage
            full_checkpoint_path = os.path.join(args.checkpoint_path,
                                                f"checkpoint.{stage_id}.pth.tar.epoch.{epoch}")
            checkpoint = torch.load(full_checkpoint_path,
                                    map_location=torch.device('cpu'))

            # iterate through all modules in stage_id's checkpoint
            local_module_id = 0

            # quit when checkpoints for all modules in full model are loaded
            while module_id < num_modules:

                # load checkpoint corresponding to different modules in our runtime
                state_dict = checkpoint["state_dict"]
                state_dict_key = "module%d" % local_module_id

                if state_dict_key not in state_dict:
                    break
                state_dict = checkpoint["state_dict"][state_dict_key]

                # remove mask buffer
                keys_to_delete = []
                for key in state_dict:
                    if "mask" in key:
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    del state_dict[key]

                if checkpoint_from_distributed(state_dict):
                    state_dict = unwrap_distributed(state_dict)

                # collect all state_dicts in a single OrderedDict
                for key in state_dict:
                    all_stages_state_dict[(stage_id, local_module_id, key)] = state_dict[key]

                stage_module, _, _ = model[module_id]
                for key in state_dict:
                    # key_to_module_mapping maps key (in state_dict) to the
                    # torch.nn.Module wrapping the parameter and the name
                    # of parameter (weight, bias, etc.)
                    key_to_module_mapping[(stage_id, local_module_id, key)] = get_submodule_and_parameter_name(
                        stage_module, key)

                # load tokenizer state
                tokenizer = Tokenizer()
                tokenizer.set_state(checkpoint['tokenizer'])
                vocab_size = tokenizer.vocab_size

                local_module_id += 1
                module_id += 1

        epoch += 1

        # build model, and load state dict
        model_config = {'vocab_size': vocab_size, 'batch_first': args.batch_first,
                        'hidden_size': 1024, 'num_layers': args.num_layers,
                        'dropout': 0.2, 'share_embedding': False}
        model = GNMT(**model_config)
        model_state_dict = OrderedDict()
        for real_key in model.state_dict():
            (module, parameter_name) = get_submodule_and_parameter_name(
                model, real_key)
            # find key in all_stages_state_dict that corresponds to real_key in
            # model's state_dict
            for key in key_to_module_mapping:
                (module2, parameter_name2) = key_to_module_mapping[key]
                if parameter_name == parameter_name2 and str(module) == str(module2):
                    break
            if parameter_name == parameter_name2 and str(module) == str(module2):
                model_state_dict[real_key] = all_stages_state_dict[key]
                del key_to_module_mapping[key]
                del all_stages_state_dict[key]

        # load state_dict into model, and perform inference
        model.load_state_dict(model_state_dict)

        if args.math == 'fp32':
            dtype = torch.FloatTensor
        if args.math == 'fp16':
            dtype = torch.HalfTensor

        model.type(dtype)
        model = model.cuda()
        model.eval()

        # construct the dataset
        test_data = TextDataset(src_fname=args.input,
                                tokenizer=tokenizer,
                                sort=False)

        # build the data loader
        test_loader = test_data.get_loader(world_size=1, rank=0,
                                           batch_size=args.batch_size,
                                           batch_first=args.batch_first,
                                           shuffle=False,
                                           pad=True,
                                           num_workers=0)

        # build the translator object
        translator = Translator(model=model,
                                tokenizer=tokenizer,
                                loader=test_loader,
                                beam_size=args.beam_size,
                                max_seq_len=args.max_seq_len,
                                len_norm_factor=args.len_norm_factor,
                                len_norm_const=args.len_norm_const,
                                cov_penalty_factor=args.cov_penalty_factor,
                                cuda=args.cuda,
                                print_freq=args.print_freq,
                                dataset_dir=args.dataset_dir)

        # execute the inference
        test_bleu, _ = translator.run(calc_bleu=args.bleu, eval_path=args.output,
                                      reference_path=args.reference, summary=True)
        print(f'{epoch}\t{test_bleu:.2f}')

if __name__ == '__main__':
    main()
