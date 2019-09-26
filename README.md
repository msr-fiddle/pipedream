# PipeDream: Generalized Pipeline Parallelism for DNN Training

This repository contains the source code implementation of the SOSP paper
"PipeDream: Generalized Pipeline Parallelism for DNN Training". This work was
done as part of Microsoft Research's [Project Fiddle](https://aka.ms/msr-fiddle). This source code
is available under the [MIT License](LICENSE.txt).

## Directory Structure

### `graph`

This contains a Python implementation of a graph, used by the PipeDream profiler
and optimizer. Profiling scripts in `profiler` generate graph profiles, that can
then be ingested by the optimizer located in `optimizer` to generate a partitioned
model, that can then be fed to the PipeDream runtime.

### `profiler`

Instrumented PyTorch applications which return profiles that can be ingested by
the optimizer.

### `optimizer`

A Python implementation of PipeDream's optimizer.

### `runtime`

PipeDream's runtime, which implements model parallelism, as well as input
pipelining in PyTorch. This can be fused with data parallelism to give hybrid
model and data parallelism, and input pipelining.

## Setup

### Software Dependencies

To run PipeDream, you will need a NVIDIA GPU with CUDA 10.0, GPU driver version 418.56, nvidia-docker2,
and Python 3. On a Linux server with NVIDIA GPU(s) and Ubuntu 16.04, these dependencies can be installed
using,

```bash
bash setup.sh
```

All dependencies are in the nvcr.io/nvidia/pytorch:19.05-py3 container, which can be downloaded using,

```bash
nvidia-docker pull nvcr.io/nvidia/pytorch:19.05-py3
```

To run the PipeDream profiler, you will need to build a new Docker image, which can be done using the
Dockerfile in this directory. Note that the Dockerfile has a dependency on the `pre_hook.patch`  and
`requirements.txt` files in this directory. This container can be built using,

```bash
docker build --tag <CONTAINER_NAME> .
```

The PyTorch Docker Container can then be run using,

```bash
nvidia-docker run -it -v /mnt:/mnt --ipc=host --net=host <CONTAINER_NAME> /bin/bash
```

### Data

#### Image Classification
All image classification experiments are run using the ImageNet ILSVC 2012 dataset.
This can be downloaded using the following command (within the docker container above),

```bash
cd scripts; python download_imagenet.py --data_dir <DATASET_DIR>
```

Note that the ImageNet dataset is about 145GB, so this download script can take some time.

#### Translation
All translation experiments are run using the WMT En-De dataset, also used for the MLPerf
translation (RNN) task. This can be downloaded using the instructions in [the MLPerf
repository](https://github.com/mlperf/training_results_v0.5/tree/master/v0.5.0/nvidia/submission/code/translation/pytorch#2-directions).


## End-to-end Workflow

To run a demo, run the following commands (the optimizer and runtime have been verified to work unchanged in `nvcr.io/nvidia/pytorch:19.05-py3`).
More detailed instructions for each of the individual components are in the corresponding directory READMEs,
and more detailed instructions on how to run the main experiments in the SOSP paper are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

[from `pipedream/profiler/image_classification`; you will need to have the changes to PyTorch listed above]
Note that the profiling step must be run with only a single GPU (hence the `CUDA_VISIBLE_DEVICES=0` before the command).

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a vgg16 -b 64 --data_dir <path to ImageNet directory>
```

[from `pipedream/optimizer`]

```bash
python optimizer_graph_hierarchical.py -f ../profiler/image_classification/profiles/vgg16/graph.txt -n 4 --activation_compression_ratio 1 -o vgg16_partitioned
```

[from `pipedream/optimizer`]

```bash
python convert_graph_to_model.py -f vgg16_partitioned/gpus=4.txt -n VGG16Partitioned -a vgg16 -o ../runtime/image_classification/models/vgg16/gpus=4 --stage_to_num_ranks 0:3,1:1
```

[from `pipedream/runtime/image_classification`; run on 4 GPUs (including a single server with 4 GPUs)]

```bash
python main_with_runtime.py --module models.vgg16.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 0 --local_rank 0 --master_addr <master IP address> --config_path models/vgg16/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 1 --local_rank 1 --master_addr <master IP address> --config_path models/vgg16/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 2 --local_rank 2 --master_addr <master IP address> --config_path models/vgg16/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 3 --local_rank 3 --master_addr <master IP address> --config_path models/vgg16/gpus=4/hybrid_conf.json --distributed_backend gloo
```

`master IP address` here is the IP address of the rank 0 process. On a server with 4 GPUs, `localhost` can be specified.

When running DP setups, please use the `nccl` backend for optimal performance. When running hybrid setups, please use
the `gloo` backend.


## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.
