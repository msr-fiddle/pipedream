# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import imagenet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path where ImageNet dataset should be downloaded")
    args = parser.parse_args()

    imagenet.parse_train_archive(args.data_dir)
    imagenet.parse_devkit_archive(args.data_dir)
    imagenet.parse_val_archive(args.data_dir)