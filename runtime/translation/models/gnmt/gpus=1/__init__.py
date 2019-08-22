# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .gnmt import GNMTGenerated 

def arch():
    return "gnmt"

def model(criterion):
    return [
        (gnmt.GNMTGenerated(), ["input0", "input1", "input2"], ["output"]),
        (criterion, ["output"], ["loss"])
    ]

def full_model():
    return GNMTGenerated()
