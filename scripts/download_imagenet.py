# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import imagenet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path where ImageNet dataset should be downloaded")
    args = parser.parse_args()

    imagenet_data = imagenet.ImageNet(args.data_dir, download=True, split='train')
    imagenet_data = imagenet.ImageNet(args.data_dir, download=True, split='val')
