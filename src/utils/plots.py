from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .misc import extract_metric_values_over_evaluation_experiences


def plot_metric_over_evaluation_experiences(
        file_path_or_buf: str | pd.DataFrame | list[str | pd.DataFrame],
        metric: str, title: str, xlabel: str, ylabel: str, grid: bool = True,
        legend: bool = True, show: bool = True, experiences: Iterable[int] = None,
        save: bool = True, savepath: str = None, num_exp: int = None,
        from_beginning: bool = True, title_size=None, xlabel_size=None, ylabel_size=None,
        legend_size=None, ylim_max=None, xlim_max=None, axes_size=16,
        base_label: str | list[str] = 'Eval Experience', linestyles: str | list[str] = '-',
        extend_label: bool = True, colors: list[list[str]] = None,
):
    """
    Plots the values of a given tracked metric on each evaluation experience data across all training experiences.
    :param file_path_or_buf: Either file path or DataFrame containing metric values.
    :param metric: Metric name.
    :param title: Title of the plot.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param grid: If True, shows grid. Default is True.
    :param legend: If True, shows legend. Default is True.
    :param show: If True, shows plot. Default is True.
    :param experiences: An Iterable representing the evaluation experiences to be used.
    :param save: If True, saves plot. Default is True.
    :param savepath: Path to save plot. Default is None.
    :param num_exp: Number of training experiences to consider for the plot.
    :param from_beginning: If True, for each evaluation experience i it plots also metric values *before*
    reaching training experience i, otherwise it plots data only from experience i. Default is True.
    :param title_size: Size of the title of the plot.
    :param xlabel_size: Size of the x-axis label of the plot.
    :param ylabel_size: Size of the y-axis label of the plot.
    :param legend_size: Size of the legend labels of the plot.
    :param ylim_max: If not None, fixes the maximum value to be shown on the y-axis. Default is None.
    :param xlim_max: If not None, fixes the maximum value to be shown on the x-axis. Default is None.
    :param axes_size: If not None, fixes the size of the numbers to be shown on the axes. Default is None.
    :param base_label: Legend will use a label of the form f"{base_label} {exp_id}" for each experience.
    :param linestyles: Linestyles, default to "-".
    :param colors: Colors of the plots.
    """
    if isinstance(file_path_or_buf, str):
        dfs: list[pd.DataFrame] = [pd.read_csv(file_path_or_buf)]
    elif isinstance(file_path_or_buf, pd.DataFrame):
        dfs: list[pd.DataFrame] = [file_path_or_buf]
    else:
        dfs: list[pd.DataFrame] = [pd.read_csv(item) if isinstance(item, str) else item for item in file_path_or_buf]
    base_labels = len(dfs) * [base_label] if isinstance(base_label, str) else base_label
    linestyles = len(dfs) * [linestyles] if isinstance(linestyles, str) else linestyles
    plt.figure(figsize=(12, 8))
    default_num_exp = len(dfs[0]['eval_exp'].unique())
    num_exp = default_num_exp if num_exp is None else num_exp
    if experiences is None:
        experiences = range(default_num_exp)
    for index, (df, label, linestyle) in enumerate(zip(dfs, base_labels, linestyles)):
        dict_data = {}
        colors_list = colors[index] if colors is not None else None
        for color_index, eval_exp in enumerate(experiences):
            value = df[df['eval_exp'] == eval_exp][metric].to_numpy()
            final_label = f"{label} {eval_exp}" if extend_label else label
            dict_data[final_label] = value
        ddf = pd.DataFrame(dict_data)
        if from_beginning:
            ddf.plot(kind='line', marker='o', linestyle=linestyle, color=colors_list)
        else:
            for eval_exp, color in zip(experiences, colors_list):
                column = f"{label} {eval_exp}" if extend_label else label
                plt.plot(
                    ddf.index[eval_exp:], ddf[column][eval_exp:], label=column,
                    marker='o', linestyle=linestyle, color=color
                )
    if ylim_max is not None:
        plt.ylim(0, ylim_max)
    if xlim_max is not None:
        plt.xlim(0, xlim_max)
    plt.tick_params(axis='both', labelsize=axes_size)
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=xlabel_size)
    plt.ylabel(ylabel, fontsize=ylabel_size)
    plt.grid(grid)
    if legend: plt.legend(fontsize=legend_size)
    if show: plt.show()
    if save: plt.savefig(savepath)
    plt.close()


def plot_metrics_over_training_experiences(
        file_path_or_buf: str | pd.DataFrame, metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, experiences: Iterable[int] = None,
        save: bool = True, savepath: str = None, num_exp: int = None, base_label: str = 'Training Experience'
):
    df = pd.read_csv(file_path_or_buf) if isinstance(file_path_or_buf, str) else file_path_or_buf
    default_num_exp = len(df['eval_exp'].unique())
    num_exp = default_num_exp if num_exp is None else num_exp
    if experiences is None:
        experiences = range(default_num_exp)
    num_epochs = len(df['epoch'].unique())
    ddf = pd.DataFrame({'epoch': np.arange(num_epochs)})
    for training_exp in experiences:
        selected_df = df[df['training_exp'] == training_exp][metric]
        ddf[f"{base_label} {training_exp}"] = selected_df
        plt.plot(ddf.index, selected_df, label=f"{base_label} {training_exp}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    if legend: plt.legend()
    if save: plt.savefig(savepath)
    if show: plt.show()


def plot_metric_over_evaluation_experiences_multiple_runs(
        file_paths_or_bufs: list[str | pd.DataFrame], metric: str, title: str, xlabel: str, ylabel: str,
        grid: bool = True, legend: bool = True, show: bool = True, experiences: Iterable[int] = None,
        save: bool = True, savepath: str = None, num_exp: int = None, from_beginning: bool = True,
        base_label: str = 'Eval Experience'
):
    dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths_or_bufs]
    default_num_exp = len(dfs[0]['eval_exp'].unique())
    num_exp = default_num_exp if num_exp is None else num_exp
    if experiences is None:
        experiences = range(default_num_exp)
    ddfs = []
    for df in dfs:
        dict_data = {}
        for eval_exp in experiences:
            value = df[df['eval_exp'] == eval_exp][metric].to_numpy()
            dict_data[f"{base_label} {eval_exp}"] = value
        ddf = pd.DataFrame(dict_data)
        ddfs.append(ddf)
    x_values = list(range(num_exp))
    plt.figure(figsize=(12, 8))
    means = {}
    stds = {}
    for column in ddfs[0].columns:
        stacked = np.vstack([ ddf[column] for ddf in ddfs ])
        means[column] = stacked.mean(axis=0)
        stds[column] = stacked.std(axis=0)
    for idx, (column, mean_values) in enumerate(means.items()):
        std_vals = stds[column]
        index = 0 if from_beginning else idx
        plt.plot(
            x_values[index:], mean_values[index:], label=column, marker='o', linestyle='-'
        )
        plt.fill_between(
            x_values[index:], (mean_values - std_vals)[index:],
            (mean_values + std_vals)[index:], alpha=0.2
        )
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.grid(grid)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if legend: plt.legend(fontsize=18)
    if show: plt.show()
    if save: plt.savefig(savepath)
    plt.close()


def plot_metric_over_multiple_strategies(
    file_paths_or_bufs: dict[str, tuple[str | pd.DataFrame, str, str]], grid: bool = True,
    legend: bool = True, show: bool = True, save: bool = True, savepath: str = None,
    title: str = None, xlabel: str = None, ylabel: str = None, include_std: bool = True
):
    plt.figure(figsize=(12, 8))
    for strategy_name, (df, color, linestyle) in file_paths_or_bufs.items():
        mean_col = [col for col in df.columns if col.startswith('Mean')][0]
        std_col = [col for col in df.columns if col.startswith('Std')][0]
        x_values = df['Experience'].values
        mean_values = df[mean_col].values
        std_values = df[std_col].values
        
        # Use strategy name as label, removing any 'Mean/Std' prefix
        label = strategy_name
        
        plt.plot(x_values, mean_values, label=label, marker='o', linestyle=linestyle, color=color)
        if include_std:
            plt.fill_between(
                x_values,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
                color=color
            )

    base_title = title if title is not None else "Performance Over Experiences"
    base_xlabel = xlabel if xlabel is not None else "Experience"
    base_ylabel = ylabel if ylabel is not None else "Metric Value"
    
    plt.title(base_title, fontsize=20)
    plt.xlabel(base_xlabel, fontsize=25)
    plt.ylabel(base_ylabel, fontsize=25)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    if grid:
        plt.grid(True, linestyle='--')
    if legend:
        plt.legend(fontsize=20)
    if save and savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


__all__ = [
    'plot_metric_over_evaluation_experiences',
    'plot_metrics_over_training_experiences',
    'plot_metric_over_evaluation_experiences_multiple_runs',
    'plot_metric_over_multiple_strategies',
]
