import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .misc import extract_metric_values_over_evaluation_experiences


def plot_metric_over_evaluation_experiences(
        file_path_or_buf: str | pd.DataFrame, metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, start_exp: int = 0, end_exp: int = -1,
        save: bool = True, savepath: str = None,
):
    df: pd.DataFrame = pd.read_csv(file_path_or_buf) if isinstance(file_path_or_buf, str) else file_path_or_buf
    num_exp = len(df['eval_exp'].unique())
    if (end_exp == -1) or (end_exp >= num_exp): end_exp = num_exp - 1
    dict_data = {}
    for eval_exp in range(start_exp, end_exp + 1):
        value = df[df['eval_exp'] == eval_exp][metric].to_numpy()
        dict_data[f"Eval Experience {eval_exp}"] = value
    ddf = pd.DataFrame(dict_data)
    plt.figure(figsize=(12, 8))
    ddf.plot(kind='line', marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    if legend: plt.legend()
    if show: plt.show()
    if save: plt.savefig(savepath)
    plt.close()


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
    dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths_or_bufs]
    num_exp = len(dfs[0]['eval_exp'].unique())
    if (end_exp == -1) or (end_exp >= num_exp): end_exp = num_exp - 1
    ddfs = []
    for df in dfs:
        dict_data = {}
        for eval_exp in range(start_exp, end_exp + 1):
            value = df[df['eval_exp'] == eval_exp][metric].to_numpy()
            dict_data[f"Eval Experience {eval_exp}"] = value
        ddf = pd.DataFrame(dict_data)
        ddfs.append(ddf)
    x_values = list(range(num_exp))
    plt.figure(figsize=(12, 8))
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
    plt.close()


def plot_forgetting_over_multiple_strategies(
    file_paths_or_bufs: dict[str, str | pd.DataFrame], experience_id: int, grid: bool = True,
    legend: bool = True, show: bool = True, save: bool = True, savepath: str = None,
):
    keys, values = [], []
    for k, v in file_paths_or_bufs.items():
        keys.append(k)
        values.append(v)
    result_dfs = extract_metric_values_over_evaluation_experiences(
        values, 'Forgetting_Exp', num_exp=10
    )
    series = [df[experience_id].to_numpy() for df in result_dfs]
    dict_data = {key: data for key, data in zip(keys, series)}
    df = pd.DataFrame(dict_data)
    df.plot(
        kind='line', figsize=(12, 8),
        title=f"Forgetting over CL Strategies on Evaluation Data of Experience {experience_id}",
        xlabel="Evaluation Experience", ylabel="Forgetting", grid=grid, legend=legend
    )
    if save: plt.savefig(savepath)
    if show: plt.show()
    plt.close()


__all__ = [
    'plot_metric_over_evaluation_experiences',
    'plot_metrics_over_training_experiences',
    'plot_metric_over_evaluation_experiences_multiple_runs',
    'plot_forgetting_over_multiple_strategies',
]
