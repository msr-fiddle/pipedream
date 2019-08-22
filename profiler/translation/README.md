# PyTorch Profiler, Translation

To run the profiler, run

```bash
python train.py \
  --dataset-dir <path to wmt_ende data> \
  --target-bleu 21.8 \
  --epochs 20 \
  --math fp32 \
  --print-freq 10 \
  --arch gnmt \
  --batch-size 64 \
  --test-batch-size 128 \
  --model-config "{'num_layers': 4, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': False}" \
  --optimization-config "{'optimizer': 'FusedAdam', 'lr': 1.75e-3}" \
  --scheduler-config "{'lr_method':'mlperf', 'warmup_iters':1000, 'remain_steps':1450, 'decay_steps':40}"
```

This will create a `gnmt` directory in `profiles/`, which will contain various
statistics about the GNMT model, including activation size, parameter size,
and forward and backward computation times, along with a serialized graph object
containing all this metadata.
