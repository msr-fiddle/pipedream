## SOSP 2019 Experiments

This document describes how to run the main experiments in the SOSP 2019 paper.
The goal of this document is to satisfy the ACM "Artifact Functional" requirements.

## System Requirements

Code in this repository will need NVIDIA GPU(s) with CUDA 10.1 and GPU driver version 418.56.
In addition, to run all experiments, nvidia-docker2 will be needed.

As described in the main README file, these dependencies can be installed using,

```bash
bash setup.sh
```

Experiments can be reproduced using the `nvcr.io/nvidia/pytorch:19.05-py3`
container, which should be pulled on all target machines using,

```bash
nvidia-docker pull nvcr.io/nvidia/pytorch:19.05-py3
```

Note that the translation experiments require a small change to the `pytorch:19.05-py3`
image, detailed below.

## Data

### Image Classification
All image classification experiments are run using the ImageNet ILSVC 2012 dataset.
This can be downloaded using the following command (within the docker container above),

```bash
cd scripts; python download_imagenet.py --data_dir <DATASET_DIR>
```

Note that the ImageNet dataset is about 145GB, so this download script can take some time.

### Translation
All translation experiments are run using the WMT En-De dataset, also used for the MLPerf
translation (RNN) task. This can be downloaded using the instructions in [the MLPerf
repository](https://github.com/mlperf/training_results_v0.5/tree/master/v0.5.0/nvidia/submission/code/translation/pytorch#2-directions).

## EC2 AMI

All required software dependencies have been installed on the following
AMI on Amazon EC2 (along with input data preparation),

| Field  | Value |
| -------------  | ------------- |
| Cloud Provider | AWS |
| Region         | us-east-2  |
| AMI ID         | ami-095caec674adbdea0  |
| AMI Name       | pipedream |

See [this link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html)
for how to find and launch a public AMI (this assumes you have a valid billable AWS account setup).

## Reproducing Experiments

Experiments can be easily launched using the driver program `driver.py` in
`runtime/`. Note that to run `driver.py`, you will need the `pyyaml` module
on Python3.

`driver.py` takes the following arguments,

```bash
python driver.py -h
usage: driver.py [-h] --config_file CONFIG_FILE [--resume PATH]
                 [--launch_single_container]
                 [--mount_directories MOUNT_DIRECTORIES [MOUNT_DIRECTORIES ...]]
                 [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Path to configuration file
  --resume PATH         path to latest checkpoint (default: none)
  --launch_single_container
                        launch a single container per machine
  --mount_directories MOUNT_DIRECTORIES [MOUNT_DIRECTORIES ...]
                        list of directories to mount
  --quiet               quiet execution
```

The most important argument here is `--config_file`, which specifies the model
and configuration to run, along with other metadata fields needed to train the
model.

An example configuration file is reproduced below,
```
[image_classification/driver_configs/vgg16_4dp.yml]

'log_directory': '/datadrive3/output_logs'
'module': 'models.vgg16.gpus=4'
'data_dir': '/mnt/data/imagenet/'
'config_file': 'models/vgg16/gpus=4/dp_conf.json'
'container': 'fiddlev3.azurecr.io/pipedream:latest'
'machines': ['localhost:0','localhost:1','localhost:2','localhost:3']
'batch_size': 64
'learning_rate': 0.01
'weight_decay': 0.0005
'epochs': 60
'print_frequency': 100
'verbose_frequency': 100
'compress_activations': False
'compress_in_gpu': False
'learning_rate_policy': 'polynomial'
'model_type': 'image_classification'
'distributed_backend': 'nccl'
```

When running your experiment, you will likely need to change `log_directory`,
`data_dir`, `container`, and `machines` fields in the relevant configuration file.

- `log_directory`: Output log directory. You will need to make
  sure all GPU workers can access this directory. In addition, you will need to
  make sure this directory is correctly mounted in the container used to run
  training, using the `--mount_directories` argument in `driver.py`. `driver.py`
  expects this directory to already be created.
- `data_dir`: Location of the input data directory.
- `container`: Name of the container used. For image classification tasks, can be
  `nvcr.io/nvidia/pytorch:19.05-py3`. For translation tasks, an addition setup
  step is needed, which is described below.
- `machines`: IP addresses of the GPUs to run the job. The number subscript indexes
  GPUs on the machine. In the above configuration file, the target worker has 4
  GPUs.

The `driver_configs` directories under `runtime/image_classification` and
`runtime/translation` have configuration files to run the optimal configuration
returned by PipeDream's optimizer (`*_(\d+)pipedream.yml`), as well as data
parallelism (`*_(\d+)dp.yml`). The multi-digit number here is the number of workers.
Please update the fields described above as appropriate to run your experiment.

For optimal performance, the `--launch_single_container` argument should
be passed in.

`--mount_directories` is needed to ensure that all directories on the host machine
that need to be visible in the running docker container are correctly mounted.
This includes the directory with source code, `log_directory`, and the directory
with input data.

`scripts/terminate_runtime.py` can be used to terminate execution of running jobs.
This can be run like,

```bash
python scripts/terminate_runtime.py <LOG_DIRECTORY>/machinefile
```

### Updating Docker Container for Translation

Translation tasks require an additional extra step to the container before running
experiments.

```bash
nvidia-docker run -it -v <directory with pipedream code>:<directory with pipedream code> --ipc=host nvcr.io/nvidia/pytorch:19.05-py3
cd <directory with pipedream code>/runtime/translation
python setup.py install
<ctrl-p ctrl-q to exit running container>
nvidia-docker ps
<look at CONTAINER_ID of running container>
nvidia-docker commit <CONTAINER_ID> <NEW_CONTAINER_NAME>
nvidia-docker kill <CONTAINER_ID>
```

This step will need to be performed on each target machine to ensure that the
container image is correct. The container image with `NEW_CONTAINER_NAME` can
be used in the `container` field in all relevant configuration files.

### Running `driver.py`

Once the configuration file has been run, experiments can be set off using commands
like the following.

For VGG-16, 4 GPUs, Data Parallelism,
```bash
python driver.py --config_file image_classification/driver_configs/vgg16_4dp.yml --launch_single_container --mount_directories /mnt /datadrive3
```

For VGG-16, 4 GPUs, PipeDream's optimal configuration,
```bash
python driver.py --config_file image_classification/driver_configs/vgg16_4pipedream.yml --launch_single_container --mount_directories /mnt /datadrive3
```

For GNMT-16, 4 GPUs, Data Parallelism,
```bash
python driver.py --config_file translation/driver_configs/gnmt_large_4dp.yml --launch_single_container --mount_directories /mnt /datadrive3
```

For GNMT-16, 4 GPUs, PipeDream's optimal configuration,
```bash
python driver.py --config_file translation/driver_configs/gnmt_large_4pipedream.yml --launch_single_container --mount_directories /mnt /datadrive3
```

Note that in all the commands above, `/mnt` contains the input data, and `/datadrive3`
contains the PipeDream source code and the `log_directory`.


### Evaluating GNMT checkpoints

To measure the BLEU score on a validation dataset, the `measure_bleu_score.py` script needs to be used
to evaluate checkpoints stored in the provided output directory. Some example uses of this command
are below.

For GNMT-16, 16 GPUs, Data Parallelism,
```bash
python compute_bleu_scores.py --num_layers 8 --dataset-dir /mnt/data/wmt_ende/ -i /mnt/data/wmt_ende/newstest2014.tok.bpe.32000.en -o /mnt/data/wmt_ende/newstest2014.tok.bpe.32000.en.translated --checkpoint_path /datadrive3/output_logs/2019-08-20T14:27:23 --bleu -r /mnt/data/wmt_ende/newstest2014.de --module models.gnmt_large.gpus=16 --num_stages 1 --cuda
```

For GNMT-16, 16 GPUs, PipeDream's optimal configuration,
```bash
python compute_bleu_scores.py --num_layers 8 --dataset-dir /mnt/data/wmt_ende/ -i /mnt/data/wmt_ende/newstest2014.tok.bpe.32000.en -o /mnt/data/wmt_ende/newstest2014.tok.bpe.32000.en.translated --checkpoint_path /datadrive3/output_logs/2019-08-19T18:22:15 --bleu -r /mnt/data/wmt_ende/newstest2014.de --module models.gnmt_large.gpus=16 --num_stages 14 --cuda
```
