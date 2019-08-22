# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import re
import torch

def sparsity(tensor, threshold):
    num_zeros = 0
    for elem in tensor:
        if threshold is None:
            if elem == 0.0:
                num_zeros += 1
        else:
            if abs(elem) < threshold:
                num_zeros += 1
    return float(num_zeros) / len(tensor)

def measure_sparsity(directory, threshold=None):
    activation_sparsities = {}
    gradient_sparsities = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'rb') as f:
            if filename.startswith('activations'):
                m = re.search(r'activations\.(\d+)\.pkl', filename)
                if m:
                    activation_id = int(m.group(1))
                tensor = torch.load(f)
                tensor = np.array(tensor.cpu().tolist()).flatten()
                activation_sparsities[activation_id] = sparsity(tensor, threshold)
            else:
                m = re.search(r'gradients\.(\d+)\.pkl', filename)
                if m:
                    gradient_id = int(m.group(1))
                tensors = torch.load(f)
                tensors = list(tensors)
                for tensor in tensors:
                    tensor = np.array(tensor.cpu().tolist()).flatten()
                    gradient_sparsities[gradient_id] = sparsity(tensor, threshold)
    return activation_sparsities, gradient_sparsities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute sparsity on logged activations and gradients')
    parser.add_argument('-d', "--directory", required=True, type=str,
                        help="Activations directory")
    parser.add_argument('-e', "--epochs", type=int, nargs='+',
                        help="List of epochs")
    parser.add_argument('-t', "--threshold", type=float, default=None,
                        help="Threshold to use while computing sparsities")
    args = parser.parse_args()

    for epoch in args.epochs:
        print("===================================")
        print("Epoch %d" % epoch)
        print("===================================")
        sub_directory = os.path.join(args.directory, str(epoch))
        activation_sparsities, gradient_sparsities = \
            measure_sparsity(sub_directory, threshold=args.threshold)
        keys = sorted(activation_sparsities.keys())
        for key in keys:
            print(key, activation_sparsities[key],
                  gradient_sparsities[keys[-1]-key])
