#!/usr/bin/env python3
# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import argparse
import sys

import numpy as np


def read_vfiles(vfiles, dataset_name=None):
    """
    Parse validation report files
    :param vfiles: list of files
    :return:
    """
    models = {}
    for vfile in vfiles:
        model_name = vfile.split("/")[-2] if "//" not in vfile \
            else vfile.split("/")[-3]
        with open(vfile, "r") as validf:
            steps = {}
            for line in validf:
                entries = line.strip().split()
                if dataset_name is not None and entries[0] != dataset_name:
                    continue
                key = int(entries[2])
                steps[key] = {}
                for i in range(3, len(entries)-1, 2):
                    name = entries[i].strip(":")
                    value = float(entries[i+1])
                    steps[key][name] = value
        models[model_name] = steps
    return models


def plot_models(models, plot_values, output_path):
    """
    Plot the learning curves for several models
    :param models:
    :param plot_values:
    :param output_path:
    :return:
    """
    # models is a dict: name -> ckpt values
    f, axes = plt.subplots(len(plot_values), len(models),
                           sharex='col', sharey='row',
                           figsize=(3*len(models), 3*len(plot_values)))
    axes = np.array(axes).reshape((len(plot_values), len(models)))

    for col, model_name in enumerate(models):
        values = {}
        # get arrays for plotting
        for step in sorted(models[model_name]):
            logged_values = models[model_name][step]
            for plot_value in plot_values:
                if plot_value not in logged_values:
                    continue
                elif plot_value not in values:
                    values[plot_value] = [[], []]
                values[plot_value][1].append(logged_values[plot_value])
                values[plot_value][0].append(step)

        for row, plot_value in enumerate(plot_values):
            axes[row][col].plot(values[plot_value][0], values[plot_value][1])
            axes[row][0].set_ylabel(plot_value)
            axes[0][col].set_title(model_name)
    plt.tight_layout()
    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(f)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path)

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("JoeyNMT Validation plotting.")
    parser.add_argument("model_dirs", type=str, nargs="+",
                        help="Model directories.")
    parser.add_argument("--plot_values", type=str, nargs="+", default=["bleu"],
                        help="Value(s) to plot. Default: bleu")
    parser.add_argument("--output_path", type=str, default="plot.pdf",
                        help="Plot will be stored in this location.")
    parser.add_argument("--dataset_name", type=str,
                        help="Name of the dataset to plot"
                        " (first column of validations.txt)")
    args = parser.parse_args()

    vfiles = [m+"/validations.txt" for m in args.model_dirs]

    models = read_vfiles(vfiles, dataset_name=args.dataset_name)
    do_exit = False
    for model, data in models.items():
        if not data:
            print('No data found for model {}.'.format(model))
            do_exit = True
    if do_exit:
        sys.exit(1)

    plot_models(models, args.plot_values, args.output_path)
