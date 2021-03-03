import argparse
import subprocess
import threading


GPT_COMMAND_TEMPLATE = """
python -m torch.distributed.launch \
    --nnodes %(nnodes)d --node_rank %(node_rank)d \
    --nproc_per_node=%(nproc_per_node)d --master_addr %(master_addr)s \
    --master_port %(master_port)d \
    pretrain_gpt2.py \
        --tensor-model-parallel-size %(tensor_model_parallel_size)d \
        --pipeline-model-parallel-size %(pipeline_model_parallel_size)d \
        --scatter-gather-tensors-in-pipeline \
        --num-layers %(num_layers)d \
        --hidden-size %(hidden_size)d \
        --num-attention-heads %(num_attention_heads)d \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --micro-batch-size %(micro_batch_size)d \
        --global-batch-size %(global_batch_size)d \
        --train-iters 50 \
        --lr-decay-iters 320 \
        --lr 0.000015 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --data-path %(data_dir)s/gpt2_text_document \
        --vocab-file %(data_dir)s/gpt2_vocab.json \
        --merge-file %(data_dir)s/gpt2_merges.txt \
        --split 1000,0,0 \
        --log-interval 1 \
        --clip-grad 1.0 \
        --fp16 \
        --DDP-impl local \
        --loss-scale 16384 \
        --apply-query-key-layer-scaling \
        --bias-gelu-fusion \
        --bias-dropout-fusion \
        --scaled-upper-triang-masked-softmax-fusion \
        --exit-interval 60000 %(command_suffix)s
    """

BERT_COMMAND_TEMPLATE = """
python -m torch.distributed.launch \
    --nnodes %(nnodes)d --node_rank %(node_rank)d \
    --nproc_per_node=%(nproc_per_node)d --master_addr %(master_addr)s \
    --master_port %(master_port)d \
    pretrain_bert.py \
        --tensor-model-parallel-size %(tensor_model_parallel_size)d \
        --pipeline-model-parallel-size %(pipeline_model_parallel_size)d \
        --scatter-gather-tensors-in-pipeline \
        --num-layers %(num_layers)d \
        --hidden-size %(hidden_size)d \
        --num-attention-heads %(num_attention_heads)d \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --micro-batch-size %(micro_batch_size)d \
        --global-batch-size %(global_batch_size)d \
        --train-iters 50 \
        --lr-decay-iters 320 \
        --lr 0.000015 \
        --min-lr 0.00001 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --data-path %(data_dir)s/bert_text_sentence \
        --vocab-file %(data_dir)s/bert_vocab.txt \
        --split 1000,0,0 \
        --log-interval 1 \
        --clip-grad 1.0 \
        --fp16 \
        --DDP-impl local \
        --loss-scale 16384 \
        --apply-query-key-layer-scaling \
        --bias-gelu-fusion \
        --bias-dropout-fusion \
        --scaled-upper-triang-masked-softmax-fusion \
        --exit-interval 60000 %(command_suffix)s
    """


class WorkerInfo(object):
    def __init__(self, ip, port=22, internal_ip=None):
        self.ip = ip
        self.port = port
        self.internal_ip = internal_ip

    def __repr__(self):
        return '%s:%s' % (self.ip, self.port)


def kill_all(workers):
    for worker in workers:
        node_ip = worker.ip
        node_port = worker.port
        subprocess.call(
            "ssh -i ~/.ssh/deepak-sigmetrics.pem -n %s -p %s -o StrictHostKeyChecking=no \"sudo pkill -9 python*\"" % (
                node_ip, node_port),
            shell=True)

def run(commands, workers, log_file_paths):
    kill_all(workers)
    def run_helper(command, worker, log_file_path):
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        with open(log_file_path, 'w') as f:
            for line in proc.stdout:
                if line.strip() == b"Exception" or b"RuntimeError" in line:
                    print("Command ran into an exception; cleaning up processes...")
                    kill_all(workers)
                    return
                f.write(line.decode())

    threads = []
    for i, (command, worker, log_file_path) in \
        enumerate(zip(commands, workers, log_file_paths)):
        thread = threading.Thread(target=run_helper,
                                  args=(command, worker, log_file_path))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    kill_all(workers)

def run_remote(runtime_cmds, workers, output_dir, mount_directories):
    launch_cmds = []
    log_file_paths = []
    for i, runtime_cmd in enumerate(runtime_cmds):
        # Put IP addresses in a list, then use them.
        if workers[0].internal_ip is not None:
            runtime_cmd = runtime_cmd.format(workers[0].internal_ip)
        else:
            runtime_cmd = runtime_cmd.format(workers[0].ip)
        docker_cmd = 'nvidia-docker run %(mount_directories)s ' \
                     '--net=host ' \
                     '--ipc=host %(container)s /bin/bash -c' % {
            "container": "nvcr.io/nvidia/pytorch:20.11-py3",
            "mount_directories":
                " ".join(["-v %s:%s" % (x, x)
                          for x in mount_directories])
        }

        log_file_path = '%s/output.log.%d' % (output_dir, i)
        log_file_paths.append(log_file_path)

        node_ip = workers[i].ip
        node_port = workers[i].port
        launch_cmd = '%s \'cd %s; %s\'' % (docker_cmd, command_line_args.code_dir,
                                           runtime_cmd)
        if node_ip != 'localhost' and node_ip != '127.0.0.1':
            launch_cmd = 'ssh -i ~/.ssh/deepak-sigmetrics.pem -n %s -p %s -o StrictHostKeyChecking=no \"%s\"' % (
                node_ip, node_port, launch_cmd)
            launch_cmds.append(launch_cmd)
            print(launch_cmd)
        print(log_file_path)
    run(launch_cmds, workers, log_file_paths)

def run_sweep(models, model_parameters, parallel_sizes,
              technique_to_command_suffix_mapping,
              micro_batch_size, global_batch_size,
              activation_recomputation):

    for model in models:
        template = None
        if model.startswith("bert"):
            template = BERT_COMMAND_TEMPLATE
        elif model.startswith("gpt"):
            template = GPT_COMMAND_TEMPLATE
        else:
            raise Exception("Invalid model!")

        for (data_parallel_size, model_parallel_size) in parallel_sizes:
            num_gpus = data_parallel_size * model_parallel_size
            (num_layers, hidden_size, num_attention_heads) = model_parameters[model]
            for technique in technique_to_command_suffix_mapping:
                num_gpus_per_worker = command_line_args.num_gpus_per_worker
                command_suffix = technique_to_command_suffix_mapping[technique]
                nproc_per_node = min(
                    num_gpus_per_worker, 
                    num_gpus)
                nnodes = max(num_gpus // num_gpus_per_worker, 1)
                if technique == 'tensor_mp':
                    tensor_model_parallel_size = model_parallel_size
                    pipeline_model_parallel_size = 1
                else:
                    tensor_model_parallel_size = 1
                    pipeline_model_parallel_size = model_parallel_size
                global_batch_size_for_technique = global_batch_size
                if technique == 'no_pipelining':
                    global_batch_size_for_technique = data_parallel_size * micro_batch_size
                runtime_cmds = []
                for node_rank in range(nnodes):
                    args = {
                        'nproc_per_node': nproc_per_node, 'nnodes': nnodes,
                        'node_rank': node_rank,
                        'master_addr': workers[0].internal_ip,
                        'master_port': 60000,
                        'tensor_model_parallel_size': tensor_model_parallel_size,
                        'pipeline_model_parallel_size': pipeline_model_parallel_size,
                        'num_layers': num_layers,
                        'hidden_size': hidden_size,
                        'num_attention_heads': num_attention_heads,
                        'micro_batch_size': micro_batch_size,
                        'global_batch_size': global_batch_size_for_technique,
                        'data_dir': command_line_args.data_dir,
                        'command_suffix': technique_to_command_suffix_mapping[technique]
                    }
                    runtime_cmd = template % args
                    runtime_cmds.append(runtime_cmd)
                # output_dir = \
                #     "logs/throughput/activation_recomputation=%s/" \
                #     "model=%s/num_gpus=%d/model_parallel_size=%d/" \
                #     "global_batch_size=%d/micro_batch_size=%d/technique=%s" % (
                #         activation_recomputation, model, num_gpus, model_parallel_size,
                #         global_batch_size, micro_batch_size,
                #         technique)
                output_dir = \
                    "logs/throughput/" \
                    "model=%s/num_gpus=%d/model_parallel_size=%d/" \
                    "global_batch_size=%d/micro_batch_size=%d/technique=%s" % (
                        model, num_gpus, model_parallel_size,
                        global_batch_size, micro_batch_size,
                        technique)
                subprocess.call("mkdir -p %s" % output_dir, shell=True)
                run_remote(runtime_cmds, workers,
                           output_dir=output_dir,
                           mount_directories=command_line_args.mount_directories)


def read_workers_file(filename):
    workers = []
    with open(filename, 'r') as f:
        for line in f:
            worker = line.strip()
            worker_info = worker.split(":")
            assert len(worker_info) == 2 or len(worker_info) == 3, worker
            internal_ip = None
            if len(worker_info) == 3:
                internal_ip = worker_info[2]
            workers.append(WorkerInfo(ip=worker_info[0],
                                      port=worker_info[1],
                                      internal_ip=internal_ip))
    return workers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mount_directories', type=str, nargs='+',
                        help='List of directories to mount')
    parser.add_argument('--num_gpus_per_worker', type=int, required=True,
                        help='Number of GPUs per worker')
    parser.add_argument('--code_dir', type=str, required=True,
                        help='Location of code on workers')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Location of data on workers')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet execution')
    command_line_args = parser.parse_args()

    workers = read_workers_file('workers.txt')

    models = ['bert_3.8b']
    model_parameters = {
        'gpt_355m': (24, 1024, 16),
        'gpt_28.3b': (2240, 1024, 16),
        'gpt_1.3b': (24, 2048, 16),
        'gpt_2.2b': (32, 2304, 32),
        'gpt_3.8b': (32, 3072, 32),
        'bert_355m': (24, 1024, 16),
        'bert_2.2b': (32, 2304, 32),
        'bert_3.8b': (32, 3072, 32),
    }
    parallel_sizes = [(4, 16)]
    for micro_batch_size in [8]:
        for global_batch_size in [256]:
            for activation_recomputation in [True]:
                technique_to_command_suffix_mapping = {
                    'pipedream_flush': '',
                    'no_pipelining': '',
                    'pipedream_2bw': '--pipeline-no-flushes',
                    'tensor_mp': '',
                    'gpipe': '--gpipe',
                }
                if activation_recomputation:
                    for key in technique_to_command_suffix_mapping:
                        technique_to_command_suffix_mapping[key] += \
                            ' --checkpoint-activations --checkpoint-num-layers 1'
                run_sweep(models, model_parameters, parallel_sizes,
                          technique_to_command_suffix_mapping,
                          micro_batch_size, global_batch_size,
                          activation_recomputation)
