# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import os
import re
import scipy.sparse
import torch


def get_sparse_size(sparse_matrix):
    # get size of a sparse matrix
    sparse_matrix.indices = sparse_matrix.indices.astype(np.uint8)
    return (sparse_matrix.data.nbytes + sparse_matrix.indptr.nbytes +
            sparse_matrix.indices.nbytes) / 1024.

def compression_ratio(tensor):
    sparse_matrix = scipy.sparse.csc_matrix(tensor)
    sparse_size = get_sparse_size(sparse_matrix)
    regular_size = sparse_matrix.toarray().nbytes / 1024.
    return regular_size / sparse_size

def measure_compression_ratio(directory):
    activation_compression_ratios = {}
    gradient_compression_ratios = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'rb') as f:
            if filename.startswith('activations'):
                m = re.search(r'activations\.(\d+)\.pkl', filename)
                if m:
                    activation_id = int(m.group(1))
                tensor = torch.load(f)
                tensor = np.array(tensor.cpu().tolist()).flatten()
                tensor_length = len(tensor)
                tensor = np.reshape(tensor, (256, tensor_length / 256))
                activation_compression_ratios[activation_id] = compression_ratio(tensor)
            else:
                m = re.search(r'gradients\.(\d+)\.pkl', filename)
                if m:
                    gradient_id = int(m.group(1))
                tensors = torch.load(f)
                tensors = list(tensors)
                for tensor in tensors:
                    tensor = np.array(tensor.cpu().tolist()).flatten()
                    tensor_length = len(tensor)
                    tensor = np.reshape(tensor, (256, tensor_length / 256))
                    gradient_compression_ratios[gradient_id] = compression_ratio(tensor)
    return activation_compression_ratios, gradient_compression_ratios


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute compression_ratio on logged activations and gradients')
    parser.add_argument('-d', "--directory", required=True, type=str,
                        help="Activations directory")
    parser.add_argument('-e', "--epochs", type=int, nargs='+',
                        help="List of epochs")
    args = parser.parse_args()

    for epoch in args.epochs:
        print("===================================")
        print("Epoch %d" % epoch)
        print("===================================")
        sub_directory = os.path.join(args.directory, str(epoch))
        activation_compression_ratios, gradient_compression_ratios = \
            measure_compression_ratio(sub_directory)
        keys = sorted(activation_compression_ratios.keys())
        for key in keys:
            print(key, activation_compression_ratios[key],
                  gradient_compression_ratios[keys[-1]-key])
