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


def cl_plots(
    db: SecureMLExperimentDB, savepath: Optional[str], filename: str = 'cl_plots.json',
    metric: str = 'R2', set_type: Literal['eval', 'test'] = 'test', *, show: bool = True,
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12,
    ticks_fontsize: int = 10, y_range: np.ndarray = np.arange(0.70, 0.94, 0.04)
):
    """
    Automatic generation of plots between different strategies for CL experiments.
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
        record = db.get_first(model_class, query_dict, )
        assert record is not None, \
            f"Check your config file for '{field}', as the corresponding query dict '{query_dict}' matches no records in the database"
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    for strategy, label in zip(strategies, labels):
        if not strategy.pop('ignore', False): # By default strategy is NOT ignored
            strategy_dict = db.get_first(Strategy, strategy)
            exp_query_dict['id_strategy'] = strategy_dict['id']
            exp_dict = db.get_first(Experiment, exp_query_dict)
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
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12, ticks_fontsize: int = 10,
    y_range: np.ndarray = np.arange(0.70, 0.94, 0.04), include_baselines: bool = True
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
        query_dict = config[field]
        record = db.get_first(model_class, query_dict)
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    if include_baselines:
        # Retrieve Strategy and Cumulative baseline
        strategy_name = config['strategy']['name']
        baseline_query_dict = exp_query_dict.copy()
        ## First, change id_general to a General config that has "mode" == "CL"
        query_general = config['general'].copy()
        query_general['mode'] = 'CL'
        record = db.get_first(General, query_general)
        assert record is not None
        baseline_query_dict['id_general'] = record['id']
        ## Then, filter out EarlyStopping since it may differ between CL and AL(CL) experiments
        baseline_query_dict.pop('id_early_stopping', None)
        ## Strategy
        baseline_query_dict['id_active_learning'] = None
        print(baseline_query_dict)
        baseline_dict = db.get_first(Experiment, baseline_query_dict)
        assert baseline_dict is not None
        baseline_logs = baseline_dict['logs']['aggregated_metrics']
        if metric in {'times', 'cumulative_times', 'num_epochs', 'time_ratios'}:
            plot_data[f"{strategy_name} Baseline"] = baseline_logs[metric]
        else:
            plot_data[f"{strategy_name} Baseline"] = baseline_logs[f"{set_type.capitalize()}_{metric}"]['mean']
        ## Cumulative (if applicable)
        if strategy_name != 'Cumulative':
            record = db.get_first(Strategy, {'name': 'Cumulative'})
            baseline_query_dict['id_strategy'] = record['id']
            baseline_dict = db.get_first(Experiment, baseline_query_dict)
            assert baseline_dict is not None
            baseline_logs = baseline_dict['logs']['aggregated_metrics']
            if metric in {'times', 'cumulative_times', 'num_epochs', 'time_ratios'}:
                plot_data[f"Cumulative Baseline"] = baseline_logs[metric]
            else:
                plot_data[f"Cumulative Baseline"] = baseline_logs[f"{set_type.capitalize()}_{metric}"]['mean']
    # Now add ActiveLearning
    for method, label in zip(al_methods, labels):
        al_query_dict = active_learnings.copy()
        al_query_dict['standard_method'] = method
        print(al_query_dict)
        record = db.get_first(ActiveLearning, al_query_dict, )
        record_id = record['id']
        exp_query_dict['id_active_learning'] = record_id
        exp_dict = db.get_first(Experiment, exp_query_dict)
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


def claea_plots_multiple_strategies(
    db: SecureMLExperimentDB, savepath: Optional[str], filename: str = 'claea_plots_single_strategy.json',
    metric: str = 'R2', set_type: Literal['eval', 'test'] = 'test', *, show: bool = True,
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12, ticks_fontsize: int = 10,
    y_range: np.ndarray = np.arange(0.70, 0.94, 0.04)
):
    with open(filename, 'r') as fp:
        config = json.load(fp)
    strategies, strategy_labels = config.pop('strategies'), config.pop('strategy_labels')
    active_learnings, method_labels = config.pop('active_learnings'), config.pop('method_labels')
    al_methods = active_learnings.pop('methods')
    assert len(strategies) == len(strategy_labels), f"\"strategies\" and \"strategy_labels\" field in {filename} do not have the same length"
    assert len(al_methods) == len(method_labels), f"\"methods\" and \"method_labels\" field in {filename} do not have the same length"
    plot_data = {}
    num_campaigns = None
    exp_query_dict = {}
    for field, model_class in zip(
        ['general', 'scenario', 'architecture', 'loss', 'optimizer', 'scheduler', 'early_stopping'],
        [General, Scenario, Architecture, Loss, Optimizer, Scheduler, EarlyStopping]
    ):
        query_dict = config[field]
        record = db.get_first(model_class, query_dict)
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    for strategy, strategy_label in zip(strategies, strategy_labels):
        if not strategy.pop('ignore', False): # By default strategy is NOT ignored
            strategy_dict = db.get_first(Strategy, strategy)
            exp_query_dict['id_strategy'] = strategy_dict['id']
        else:
            continue
        # Now add ActiveLearning
        for method, method_label in zip(al_methods, method_labels):
            al_query_dict = active_learnings.copy()
            al_query_dict['standard_method'] = method
            record = db.get_first(ActiveLearning, al_query_dict, )
            record_id = record['id']
            exp_query_dict['id_active_learning'] = record_id
            exp_dict = db.get_first(Experiment, exp_query_dict)
            exp_logs = exp_dict['logs']['aggregated_metrics']
            label = f"{strategy_label} - {method_label}"
            if metric in {'times', 'cumulative_times', 'num_epochs', 'time_ratios'}:
                plot_data[label] = exp_logs[metric]
            else:
                plot_data[label] = exp_logs[f"{set_type.capitalize()}_{metric}"]['mean']
            print(f"Considering for ({strategy}, {method}) the experiment {exp_dict['name']} with data: {plot_data[label]}")
    
    x_values = np.arange(1, num_campaigns + 1)
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
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--metric', type=str, default='R2')
    parser.add_argument('--set-type', type=str, default='test')
    parser.add_argument('--legend-size', type=int, default=12)
    parser.add_argument('--xlabel-size', type=int, default=14)
    parser.add_argument('--ylabel-size', type=int, default=14)
    parser.add_argument('--ticks-size', type=int, default=10)
    parser.add_argument('--y-start', type=float, default=0.70)
    parser.add_argument('--y-stop', type=float, default=0.94)
    parser.add_argument('--y-step', type=float, default=0.04)
    parser.add_argument('--include-baselines', type=int, default=1)
    # Parse args
    args = parser.parse_args()
    mode, config, metric, set_type = args.mode.upper().replace('_', '-'), args.config, args.metric, args.set_type
    if config is None:
        match mode:
            case 'CL':
                config = 'cl_plots.json'
            case 'CLAEA' | 'CLAEA-SINGLE' | 'CLAEA-SINGLE-STRATEGY':
                config = 'claea_plots_single_strategy.json'
            case 'CLAEA-MULTIPLE' | 'CLAEA-MULTIPLE-STRATEGIES':
                config = 'claea_plots_multiple_strategies.json'
    match mode:
        case 'CL':
            cl_plots(
                db, savepath=args.savepath, filename=config, metric=metric, set_type=set_type,
                show=True, legend_fontsize=args.legend_size, xlabel_size=args.xlabel_size,
                ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
                y_range=np.arange(args.y_start, args.y_stop, args.y_step)
            )
        case 'CLAEA' | 'CLAEA-SINGLE' | 'CLAEA-SINGLE-STRATEGY':
            claea_plots_single_strategy(
                db, savepath=args.savepath, filename=config, metric=metric, set_type=set_type,
                show=True, legend_fontsize=args.legend_size, xlabel_size=args.xlabel_size,
                ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
                y_range=np.arange(args.y_start, args.y_stop, args.y_step),
                include_baselines=args.include_baselines
            )
        case 'CLAEA-MULTIPLE' | 'CLAEA-MULTIPLE-STRATEGIES':
            claea_plots_multiple_strategies(
                db, savepath=args.savepath, filename=config, metric=metric, set_type=set_type,
                show=True, legend_fontsize=args.legend_size, xlabel_size=args.xlabel_size,
                ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
                y_range=np.arange(args.y_start, args.y_stop, args.y_step)
            )
        case _:
            raise ValueError(f"Unknown mode '{mode}'")
