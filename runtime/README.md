# PipeDream Runtime

This directory contains implementation for the distributed runtime that integrates
model parallelism, pipelining, and data parallelism into PyTorch.

`runtime.py`: Contains the main `StageRuntime` class.

`communication.py`: Simple communication library that sends PyTorch tensors between
a single sender and receiver.

`tests`: Contains a simple test harness for the `send_tensor` and `receive_tensor`
functions in `communication.py`.

`models`: Contains implementations of models that can be run with the runtime.

`driver_configs`: Contains driver configuration files to use with `driver.py`

## Auto-generated model with runtime

`main_with_runtime.py` is a driver program for ImageNet
image classification models that uses our `StageRuntime` and integrates
with PyTorch. The runtime allows a model's layers to be split over
multiple machines, and supports pipelining.

### Using `driver.py`

`driver.py` configures containers, launches `main_with_runtime.py` within
the containers, and logs experimental settings and output.
It uses a user provided Yaml file to configure the settings:

```bash
python driver.py --config_file driver_configs/resnet50_single_machine.yml
```

All the options described below can be configured to be launched using
`driver.py`.

### Using `StageRuntime` on single machine

To use the `StageRuntime` implemented in `runtime.py` on a single
machine, use command line arguments like below.

```bash
python main_with_runtime.py --module models.resnet50.gpus=2 -b 128 --data_dir ../../../data/imagenet
```

### Using `StageRuntime` with Model Parallelism

To split the generated ResNet50 model over two machines (modules 1 & 2
on machine 1, and modules 3, 4 & 5 (loss) on machine 2) using the
`StageRuntime` implemented in `../../runtime.py`, use command line
arguments like below (`--rank`, `--master_addr`, and `--config_path` are
important).

With input pipelining,

```bash
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 0 --local_rank 0 --master_addr localhost --config_path models/resnet50/gpus=2/mp_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 1 --local_rank 1 --master_addr localhost --config_path models/resnet50/gpus=2/mp_conf.json --distributed_backend gloo
```

Without input pipelining,

```bash
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 0 --local_rank 0 --master_addr localhost --config_path models/resnet50/gpus=2/mp_conf.json --no_input_pipelining --distributed_backend gloo
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 1 --local_rank 1 --master_addr localhost --config_path models/resnet50/gpus=2/mp_conf.json --no_input_pipelining --distributed_backend gloo
```

With data parallelism (and no input pipelining),

```bash
python main_with_runtime.py --module models.resnet50.gpus=2 -b 128 --data_dir ../../../data/imagenet --rank 0 --local_rank 0 --master_addr localhost --config_path models/resnet50/gpus=2/dp_conf.json --no_input_pipelining --distributed_backend nccl
python main_with_runtime.py --module models.resnet50.gpus=2 -b 128 --data_dir ../../../data/imagenet --rank 1 --local_rank 1 --master_addr localhost --config_path models/resnet50/gpus=2/dp_conf.json --no_input_pipelining --distributed_backend nccl
```

Note that for DP-only setups, we use the `nccl` backend for optimal performance.


With hybrid parallelism (model and data parallelism, and pipelining),

```bash
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 0 --local_rank 0 --master_addr localhost --config_path models/resnet50/gpus=2/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.resnet50.gpus=2 -b 64 --data_dir ../../../data/imagenet --rank 1 --local_rank 1 --master_addr localhost --config_path models/resnet50/gpus=2/hybrid_conf.json --distributed_backend gloo
```
