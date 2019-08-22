# PipeDream Profiler

This directory contains code to profile training for different PyTorch models. `profiles` contains
some generated profiles for different models (which can be consumed by the
PipeDream++ optimizer).

Instructions for each application are in the individual READMEs.

## Directory structure


### `torchmodules`

This directory contains some user-written modules to assist with PyTorch training and inference.

- `torchgraph` determines the dependency structure of a PyTorch model and creates an instance of `Graph`
  that maps these dependencies, using a `TensorWrapper` object.
- `torchlogger` logs the outputs of the forward and backward pass of a PyTorch model at a prescribed
  frequency.
- `torchprofiler` determines the time spent in each layer of the PyTorch model.
- `torchsummary` determines the size of the outputs of each layer, as well as the number of parameters
  in each layer.
  
### Instrumented applications for each application type

`image_classification` and `translation` contain instrumented applications
with the PipeDream profiler. They return profiles for each model, annotated with output activation sizes,
parameter sizes, and compute times for each layer.

Instructions on how to use each type of instrumented application are in the respective directory's README.
