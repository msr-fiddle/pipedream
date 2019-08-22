# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse


def process_file(filename):
    with open(filename, 'r') as f:
        lines = []
        earliest_start_time = {}
        latest_end_time = {}
        for line in f:
            lines.append(line)
            line = line.strip()
            if line == "":
                continue
            if not (line.startswith("Start") or line.startswith("End")):
                pass
            elif line.startswith("Start time"):
                try:
                    time = float(line.split(": ")[1])
                    iteration = int(line.split("[")[1].split("]")[0])
                    if iteration not in earliest_start_time:
                        earliest_start_time[iteration] = time
                except:
                    continue
            elif line.startswith("End time"):
                try:
                    time = float(line.split(": ")[1])
                    iteration = int(line.split("[")[1].split("]")[0])
                    latest_end_time[iteration] = time
                except:
                    continue

    iterations = sorted(earliest_start_time.keys())
    times = []
    for iteration in iterations:
        times.append(latest_end_time[iteration] - earliest_start_time[iteration])
    print("Number of iterations: %d" % len(iterations))
    print("Average all_reduce time: %f seconds/iteration" % (sum(times) / len(times)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse reduce_log files to get per-iteration reduction times')
    parser.add_argument('-f', "--filename", required=True, type=str,
                        help="Filename to extract reduction times from")
    args = parser.parse_args()
    process_file(args.filename)
