import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_evaluation_experiences(
        file_path_or_buf: str | pd.DataFrame, metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, start_exp: int = 0, end_exp: int = -1,
        save: bool = True, savepath: str = None,
):
    df = pd.read_csv(file_path_or_buf) if isinstance(file_path_or_buf, str) else file_path_or_buf
    num_exp = len(df['eval_exp'].unique())
    print(num_exp)
    data = [[] for _ in range(num_exp)]
    for training_exp in range(num_exp):
        for eval_exp in range(num_exp):
            value = df[(df['training_exp'] == training_exp) & (df['eval_exp'] == eval_exp)][metric].iloc[0]
            data[eval_exp].append(value)
    dict_data = {}
    for i in range(num_exp):
        dict_data[f"Eval Experience {i}"] = np.array(data[i])
    dict_data = {}
    for i in range(num_exp):
        dict_data[f"Eval Experience {i}"] = np.array(data[i])
    ddf = pd.DataFrame(dict_data)
    x_values = list(range(num_exp))
    plt.figure(figsize=(12, 8))
    if (end_exp == -1) or (end_exp >= num_exp): end_exp = num_exp - 1
    for column in ddf.columns[start_exp:end_exp + 1]:
        plt.plot(x_values, ddf[column], marker='o', linestyle='-', label=column)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    if legend: plt.legend()
    if show: plt.show()
    if save: plt.savefig(savepath)


def plot_metrics_over_training_experiences(
        file_path_or_buf: str | pd.DataFrame, metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, start_exp: int = 0, end_exp: int = -1,
        save: bool = True, savepath: str = None,
):
    df = pd.read_csv(file_path_or_buf) if isinstance(file_path_or_buf, str) else file_path_or_buf
    num_exp = len(df['training_exp'].unique())
    num_epochs = len(df['epoch'].unique())
    print(num_exp, num_epochs)
    ddf = pd.DataFrame({'epoch': np.arange(num_epochs)})
    if (end_exp == -1) or (end_exp >= num_exp): end_exp = num_exp - 1
    for training_exp in range(start_exp, end_exp + 1):
        selected_df = df[df['training_exp'] == training_exp][metric]
        ddf[f"Training Exp {training_exp}"] = selected_df
        plt.plot(np.arange(num_epochs), selected_df, label=f"Training Exp {training_exp}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    if legend: plt.legend()
    if save: plt.savefig(savepath)
    if show: plt.show()


def plot_metric_over_evaluation_experiences_multiple_runs(
        file_paths_or_bufs: list[str | pd.DataFrame], metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, start_exp: int = 0, end_exp: int = -1,
        save: bool = True, savepath: str = None,
):
    dfs = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths_or_bufs]
    num_exp = len(dfs[0]['eval_exp'].unique())
    print(num_exp)
    ddfs = []
    for df in dfs:
        data = [[] for _ in range(num_exp)]
        for training_exp in range(num_exp):
            for eval_exp in range(num_exp):
                value = df[(df['training_exp'] == training_exp) & (df['eval_exp'] == eval_exp)][metric].iloc[0]
                data[eval_exp].append(value)
        dict_data = {}
        for i in range(num_exp):
            dict_data[f"Eval Experience {i}"] = np.array(data[i])
        dict_data = {}
        for i in range(num_exp):
            dict_data[f"Eval Experience {i}"] = np.array(data[i])
        ddf = pd.DataFrame(dict_data)
        ddfs.append(ddf)
    x_values = list(range(num_exp))
    plt.figure(figsize=(12, 8))
    if (end_exp == -1) or (end_exp >= num_exp): end_exp = num_exp - 1
    means = {}
    stds = {}
    for column in ddfs[0].columns[start_exp:end_exp + 1]:
        stacked = np.vstack([ ddf[column] for ddf in ddfs ])
        means[column] = stacked.mean(axis=0)
        stds[column] = stacked.std(axis=0)
    for column, mean_values in means.items():
        std_vals = stds[column]
        plt.plot(
            x_values, mean_values, label=column, marker='o', linestyle='-'
        )
        plt.fill_between(
            x_values, mean_values - std_vals, mean_values + std_vals, alpha=0.2
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    if legend: plt.legend()
    if show: plt.show()
    if save: plt.savefig(savepath)



__all__ = [
    'plot_metric_over_evaluation_experiences',
    'plot_metrics_over_training_experiences',
    'plot_metric_over_evaluation_experiences_multiple_runs',
]
