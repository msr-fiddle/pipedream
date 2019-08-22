# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def dump_stats_to_csv(layer_timestamps, iteration_timestamps, data_timestamps,
                      optimizer_step_timestamps=[], output_filename="stats.csv"):

	with open(output_filename, "w") as csvfile:
		blanks = []

		row = "iteration"
		for timestamp in iteration_timestamps:
			row += "," + str(timestamp['duration'])
			blanks.append(timestamp['duration'])
		csvfile.write(row + "\n")

		for i in range(0, len(layer_timestamps[0])):
			layer_type = str(layer_timestamps[0][i][0])
			layer_type = layer_type.replace(",", "_")
			forward_name = layer_type + " [forward]"
			backward_name = layer_type + " [backward]"
			fw_row = forward_name
			bw_row = backward_name
			for j in range(0, len(layer_timestamps)):
				layer_type = str(layer_timestamps[0][i][0])
				forward_duration = layer_timestamps[j][i][2]
				backward_duration = layer_timestamps[j][i][5]
				fw_row += "," + str(forward_duration)
				bw_row += "," + str(backward_duration)
				blanks[j] -= forward_duration
				blanks[j] -= backward_duration
			csvfile.write(fw_row + "\n")
			csvfile.write(bw_row + "\n")

		row = "data_loading"
		for j in range(0, len(data_timestamps)):
			timestamp = data_timestamps[j]
			row += "," + str(timestamp['duration'])
			blanks[j] -= timestamp['duration']
		csvfile.write(row + "\n")

		row = "optimizer_step"
		for j in range(0, len(optimizer_step_timestamps)):
			timestamp = optimizer_step_timestamps[j]
			row += "," + str(timestamp['duration'])
			blanks[j] -= timestamp['duration']
		csvfile.write(row + "\n")

		row = "gap"
		for blank in blanks:
			row += "," + str(blank)
		csvfile.write(row + "\n")
