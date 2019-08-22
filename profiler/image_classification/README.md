# PyTorch Profiler, Image Classification

To run the profiler, run

```bash
python main.py -a resnet50 -b 128 --data_dir <path to ImageNet data>
```

This will create a `resnet50` directory in `profiles/`, which will contain various
statistics about the ResNet-50 model, including activation size, parameter size,
and forward and backward computation times, along with a serialized graph object
containing all this metadata.
