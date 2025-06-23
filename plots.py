# [NEW] Plotting functions
# 1) CL with different strategies
# 2) CLAEA with single strategy and multiple methods (+ Naive and Cumulative Baselines)
# 3) CLAEA with different strategies (+ Naive and Cumulative Baselines)
from typing import Optional, Literal
import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src.utils import *
from src.database import *


def column_to_label(column: str):
    if column in ("R2", "R2Score", "R2Score_Exp"):
        return r"$R^2$"
    elif column in ("RD", "RelativeDistance", "RelativeDistance_Exp"):
        return r"$RD$"
    elif column == "R":
        return r"$R$"
    elif column == "times":
        return r"$t_i$"
    elif column == "cumulative_times":
        return r"$t_{tot,\:i}$"
    elif column == "time_ratios":
        return r"$T_i = \dfrac{t_{tot,\:i}}{t_{tot,\:Naive,\:i}}$"
    elif column == "num_epochs":
        return r"$E_i$"
    else:
        return column


def cl_plots(
    db: SecureMLExperimentDB, savepath: Optional[str], filename: str = 'cl_plots.json',
    metric: str = 'R2', set_type: Literal['eval', 'test'] = 'test', *, show: bool = True,
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12,
    ticks_fontsize: int = 10, y_range: np.ndarray = np.arange(0.70, 0.94, 0.04)
):
    """
    Authomatic generation of plots between different strategies for CL experiments.
    Config file is of the form:
    {
        "strategies": [
            {
                "name": <strategy_name>,
                "from_scratch": <is_from_scratch>,
                "ignore": <bool> # as for running experiments,
                "parameters: <parameters_dict>
            },
            ...
        ],
        "labels": [
            <label_for_first_strategy>, # whether ignored or not
            ...
        ]
    }
    """
    with open(filename, 'r') as fp:
        config = json.load(fp)
    strategies, labels = config.pop('strategies'), config.pop('labels')
    assert len(strategies) == len(labels), f"\"strategies\" and \"labels\" field in {filename} do not have the same length"
    plot_data = {}
    num_campaigns = None
    exp_query_dict = {}
    for field, model_class in zip(
        ['general', 'scenario', 'architecture', 'loss', 'optimizer', 'scheduler', 'early_stopping'],
        [General, Scenario, Architecture, Loss, Optimizer, Scheduler, EarlyStopping]
    ):
        query_dict = config[field]
        record = db.read_record_where(model_class, query_dict, as_dict=True)
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    for strategy, label in zip(strategies, labels):
        if not strategy.pop('ignore', False): # By default strategy is NOT ignored
            strategy_dict = db.read_record_where(Strategy, strategy)
            exp_query_dict['id_strategy'] = strategy_dict['id']
            exp_dict = db.read_record_where(Experiment, exp_query_dict)
            exp_logs = exp_dict['logs']['aggregated_metrics']
            if metric in {'times', 'cumulative_times', 'num_epochs', 'time_ratios'}:
                plot_data[label] = exp_logs[metric]
            else:
                plot_data[label] = exp_logs[f"{set_type.capitalize()}_{metric}"]['mean']
            print(f"Considering for {strategy} the experiment {exp_dict['name']} with data: {plot_data[label]}")
    
    x_values = np.arange(1, num_campaigns + 1)
    #plt.figure(figsize=(12, 12))
    for label, points in plot_data.items():
        plt.plot(x_values, points, label=label, marker='o', linestyle='-')
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=legend_fontsize)
    plt.xlabel(r"Experimental Campaign ($i$)", fontsize=xlabel_size)
    plt.ylabel(column_to_label(metric), fontsize=ylabel_size)
    plt.xticks(x_values, fontsize=ticks_fontsize)
    plt.yticks(y_range, fontsize=ticks_fontsize)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()


def claea_plots_single_strategy(
    db: SecureMLExperimentDB, savepath: Optional[str], filename: str = 'claea_plots_single_strategy.json',
    metric: str = 'R2', set_type: Literal['eval', 'test'] = 'test', *, show: bool = True,
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12,
    ticks_fontsize: int = 10, y_range: np.ndarray = np.arange(0.70, 0.94, 0.04)
):
    with open(filename, 'r') as fp:
        config = json.load(fp)
    active_learnings, labels = config.pop('active_learnings'), config.pop('labels')
    al_methods = active_learnings.pop('methods')
    assert len(al_methods) == len(labels), f"\"methods\" and \"labels\" field in {filename} do not have the same length"
    plot_data = {}
    num_campaigns = None
    exp_query_dict = {}
    for field, model_class in zip(
        ['general', 'scenario', 'architecture', 'loss', 'optimizer', 'scheduler', 'early_stopping', 'strategy'],
        [General, Scenario, Architecture, Loss, Optimizer, Scheduler, EarlyStopping, Strategy]
    ):
        print(field)
        query_dict = config[field]
        print(query_dict)
        record = db.read_record_where(model_class, query_dict)
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    for method, label in zip(al_methods, labels):
        al_query_dict = active_learnings.copy()
        al_query_dict['standard_method'] = method
        print(al_query_dict)
        record = db.read_record_where(ActiveLearning, al_query_dict, as_dict=True)
        record_id = record['id']
        exp_query_dict['id_active_learning'] = record_id
        exp_dict = db.read_record_where(Experiment, exp_query_dict)
        exp_logs = exp_dict['logs']['aggregated_metrics']
        if metric in {'times', 'cumulative_times', 'num_epochs', 'time_ratios'}:
            plot_data[label] = exp_logs[metric]
        else:
            plot_data[label] = exp_logs[f"{set_type.capitalize()}_{metric}"]['mean']
        print(f"Considering for {method} the experiment {exp_dict['name']} with data: {plot_data[label]}")
    
    x_values = np.arange(1, num_campaigns + 1)
    #plt.figure(figsize=(12, 12))
    for label, points in plot_data.items():
        plt.plot(x_values, points, label=label, marker='o', linestyle='-')
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=legend_fontsize)
    plt.xlabel(r"Experimental Campaign ($i$)", fontsize=xlabel_size)
    plt.ylabel(column_to_label(metric), fontsize=ylabel_size)
    plt.xticks(x_values, fontsize=ticks_fontsize)
    plt.yticks(y_range, fontsize=ticks_fontsize)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()


if __name__ == '__main__':
    db = get_db()
    # Arg parser definition
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='CL')
    parser.add_argument('--config', type=str, default='cl_plots.json')
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--metric', type=str, default='R2')
    parser.add_argument('--set-type', type=str, default='test')
    parser.add_argument('--legend-size', type=int, default=12)
    parser.add_argument('--xlabel-size', type=int, default=14)
    parser.add_argument('--ylabel-size', type=int, default=14)
    parser.add_argument('--ticks-size', type=int, default=16)
    parser.add_argument('--y-start', type=float, default=0.70)
    parser.add_argument('--y-stop', type=float, default=0.94)
    parser.add_argument('--y-step', type=float, default=0.04)
    # Parse args
    args = parser.parse_args()
    mode, config, metric, set_type = args.mode.upper(), args.config, args.metric, args.set_type
    match mode:
        case 'CL':
            cl_plots(
                db, savepath=args.savepath, filename=config, metric=metric, set_type=set_type,
                show=True, legend_fontsize=args.legend_size, xlabel_size=args.xlabel_size,
                ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
                y_range=np.arange(args.y_start, args.y_stop, args.y_step)
            )
        case 'CLAEA' | 'CLAEA-SINGLE-STRATEGY':
            claea_plots_single_strategy(
                db, savepath=args.savepath, filename=config, metric=metric, set_type=set_type,
                show=True, legend_fontsize=args.legend_size, xlabel_size=args.xlabel_size,
                ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
                y_range=np.arange(args.y_start, args.y_stop, args.y_step)
            )
        case 'CLAEA-MULTIPLE-STRATEGIES':
            raise NotImplementedError(f"{mode} temporarily not implemented")
        case _:
            raise ValueError(f"Unknown mode '{mode}'")
