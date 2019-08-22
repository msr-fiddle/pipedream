# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .vgg16 import VGG16Partitioned

def arch():
    return "vgg16"

def model(criterion):
    return [
        (VGG16Partitioned(), ["input"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]

def full_model():
    return VGG16Partitioned()
