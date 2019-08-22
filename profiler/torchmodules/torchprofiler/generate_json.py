# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

def generate_json_helper(start, end, timestamps, name_base):
    json_blobs = []
    for i in range(start, end+1):
        json_blob = {}
        json_blob['pid'] = i
        json_blob['tid'] = timestamps[i]['pid']
        json_blob['ts'] = timestamps[i]['start']
        json_blob['dur'] = timestamps[i]['duration']
        json_blob['ph'] = 'X'
        json_blob['name'] = name_base + (" (iteration %d)" % i)
        json_blobs.append(json_blob)
    return json_blobs

def generate_json(start, end, layer_timestamps, iteration_timestamps, data_timestamps,
                  optimizer_step_timestamps=[], output_filename="processed_time.json"):
    json_blob = {"traceEvents": []}

    for i in range(0, len(layer_timestamps[0])):
        for j in range(0, len(layer_timestamps)):
            layer_type = str(layer_timestamps[0][i][0])
            forward_name = layer_type + (" [forward; iteration %d]" % j)
            forward_pid = layer_timestamps[j][i][3]
            forward_start = layer_timestamps[j][i][1]
            forward_duration = layer_timestamps[j][i][2]
            backward_name = layer_type + (" [backward; iteration %d]" % j)
            backward_pid = layer_timestamps[j][i][6]
            backward_start = layer_timestamps[j][i][4]
            backward_duration = layer_timestamps[j][i][5]

            if not (start <= j <= end):
                continue

            json_blob['traceEvents'].append({'name': forward_name,
                                             'ph': 'X',
                                             'pid': j,
                                             'tid': forward_pid, 
                                             'ts': forward_start * 1000, 
                                             'dur': forward_duration * 1000,
                                             'iteration': j})
            json_blob['traceEvents'].append({'name': backward_name,
                                             'ph': 'X',
                                             'pid': j,
                                             'tid': backward_pid,
                                             'ts': backward_start * 1000,
                                             'dur': backward_duration * 1000,
                                             'iteration': j})

    for i in range(start, end+1):
        blob = {'pid': 0, 'tid': 1}
        blob['ts'] = iteration_timestamps[i]['start']
        blob['dur'] = iteration_timestamps[i]['duration']
        blob['ph'] = 'X'
        blob['name'] = "Iteration %d" % i
        json_blob['traceEvents'].append(blob)

    json_blob['traceEvents'].extend(generate_json_helper(start, end, data_timestamps, "data_loading"))
    if optimizer_step_timestamps != []:
        json_blob['traceEvents'].extend(generate_json_helper(start, end, optimizer_step_timestamps,
                                                             "optimizer.step()"))

    json.dump(json_blob, open(output_filename, 'w'),
              sort_keys=True, indent=4, separators=(',', ': '))
