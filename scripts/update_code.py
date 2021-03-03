import subprocess

class WorkerInfo(object):
    def __init__(self, ip, port=22, internal_ip=None):
        self.ip = ip
        self.port = port
        self.internal_ip = internal_ip

    def __repr__(self):
        return '%s:%s' % (self.ip, self.port)


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

def kill_all(workers):
    for worker in workers:
        node_ip = worker.ip
        node_port = worker.port
        subprocess.call(
            "ssh -n %s -p %s -o StrictHostKeyChecking=no 'sudo rm -rf megatron'" % (
                node_ip, node_port),
            shell=True)
        subprocess.call(
            "rsync -rv ../megatron %s:/home/ubuntu/" % (
                node_ip),
            shell=True)


if __name__ == '__main__':
    workers = read_workers_file('workers.txt')
    kill_all(workers)
