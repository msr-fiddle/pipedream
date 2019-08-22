# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import re


def read_file(filename):
    activation_values = {}
    gradient_values = {}
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            m = re.search(r'Epoch (\d+)', line)
            if m:
                epoch = int(m.group(1))
            m = re.search(r'(\d+) (.*) (.*)', line)
            if m:
                if epoch not in activation_values:
                    activation_values[epoch] = []
                if epoch not in gradient_values:
                    gradient_values[epoch] = []
                id = int(m.group(1))
                activation_values[epoch].append(float(m.group(2)))
                gradient_values[epoch].append(float(m.group(3)))
                assert(len(activation_values[epoch]) == id+1)
                assert(len(gradient_values[epoch]) == id+1)
    return activation_values, gradient_values


def plot(values, epochs_to_plot, ylimit, ylabel, output_filepath):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    matplotlib.rc('text', usetex=True)
    sns.set_style('ticks')
    sns.set_style({'font.family':'sans-serif'})
    flatui = ['#002A5E', '#FD151B', '#8EBA42', '#348ABD', '#988ED5', '#777777', '#8EBA42', '#FFB5B8']
    sns.set_palette(flatui)
    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 10}
    sns.set_context("paper", font_scale=3,  rc=paper_rc)
    current_palette = sns.color_palette()

    plt.figure(figsize=(10, 4))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    for epoch_to_plot in epochs_to_plot:
        values_to_plot = values[epoch_to_plot]
        ax.plot(range(len(values_to_plot)), values_to_plot,
                label="Epoch %d" % epoch_to_plot,
                linewidth=2)
    ax.set_xlim([0, None])
    ax.set_ylim([0, ylimit])

    ax.set_xlabel("Layer ID")
    ax.set_ylabel(ylabel)
    plt.legend()

    with PdfPages(output_filepath) as pdf:
        pdf.savefig(bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot computed values on logged activations and gradients')
    parser.add_argument('-f', "--input_filename", required=True, type=str,
                        help="Input filename")
    parser.add_argument('-e', "--epochs", type=int, nargs='+',
                        help="List of epochs to plot")
    parser.add_argument('-o', "--output_directory", required=True, type=str,
                        help="Output directory")
    parser.add_argument('-y', "--ylabel", required=True, type=str,
                        help="y-label for plot")
    parser.add_argument("--ylimit", default=None, type=float,
                        help="Limit of y-axis")
    args = parser.parse_args()

    activation_values, gradient_values = read_file(args.input_filename)
    try:
        os.mkdir(args.output_directory)
    except:
        pass
    plot(activation_values, args.epochs, args.ylimit, args.ylabel, os.path.join(args.output_directory, "activations.pdf"))
    plot(gradient_values, args.epochs, args.ylimit, args.ylabel, os.path.join(args.output_directory, "gradients.pdf"))
