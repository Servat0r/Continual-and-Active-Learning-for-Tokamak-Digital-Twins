import os
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


__all__ = ['plot_metric_over_evaluation_experiences']
