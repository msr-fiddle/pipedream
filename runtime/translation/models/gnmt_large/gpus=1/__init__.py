# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .gnmt_large import GNMTGenerated

def arch():
    return "gnmt_large"

def model(criterion):
    return [
        (gnmt_large.GNMTGenerated(), ["input0", "input1", "input2"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]

def full_model():
    return GNMTGenerated()
