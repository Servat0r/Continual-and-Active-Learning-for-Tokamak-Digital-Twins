import os
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.misc import column_to_label, stdout_debug_print
from ..utils.plots import plot_metric_over_evaluation_experiences


def evaluation_experiences_plots(log_folder, metric_list, title_list, ylabel_list):
    for metric, title, ylabel in zip(metric_list, title_list, ylabel_list):
        # Experiences 0-4
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, experiences=range(5),
            savepath=os.path.join(log_folder, f'plot_of_first_5_experiences_{metric[:-4]}.png'),
            from_beginning=False,
        )
        # Plot over all experiences
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, experiences=range(10),
            savepath=os.path.join(log_folder, f'plot_of_all_10_experiences_{metric[:-4]}.png'),
            from_beginning=False,
        )


def plot_training(
    log_folder: str, metric: str = 'R2Score', train_filename: str = 'training_results_epoch.csv',
    eval_filename: str = 'eval_results_epoch.csv'
):
    train_df = pd.read_csv(os.path.join(log_folder, train_filename), usecols=range(7))
    eval_df = pd.read_csv(os.path.join(log_folder, eval_filename), usecols=range(7))
    num_exps = len(train_df['training_exp'].unique())
    nrows = int(sqrt(num_exps)) # e.g. sqrt(10) = 3 + eps > 0 => We will try 3, 2 etc, and find nrows=2, ncolumns=5
    ncolumns = None
    while (nrows >= 1) and (ncolumns is None):
        if num_exps % nrows == 0:
            ncolumns = num_exps // nrows
        else:
            nrows -= 1
    #fig, ax = plt.subplots(nrows=nrows, ncols=ncolumns)
    # Adjust figsize for each internal figure
    figsize = (6 * ncolumns, 8 * nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=figsize)
    for k in range(num_exps):
        i, j = k // ncolumns, k % ncolumns
        train_data = train_df[train_df.training_exp == k][f"{metric}_Epoch"].to_numpy()
        eval_data = eval_df[eval_df.training_exp == k][f"{metric}_Epoch"].to_numpy()
        metric_name = column_to_label(metric)
        target = ax[j] if nrows == 1 else ax[i, j]
        epochs = np.arange(1, len(train_data) + 1)
        target.plot(epochs, train_data, label=f"Training {metric_name}", marker='o', linestyle='-')
        epochs = np.arange(1, len(eval_data) + 1)
        target.plot(epochs, eval_data, label=f"Validation {metric_name}", marker='o', linestyle='-')
        target.grid(True, alpha=0.5)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    savepath = os.path.join(log_folder, f"training_plots_{metric}.pdf")
    plt.savefig(savepath)


__all__ = ['evaluation_experiences_plots', 'plot_training']