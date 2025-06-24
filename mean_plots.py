# Plots mean (arithmetic or weighted) across all experiences for a given experiment (CL-only mode)
from typing import Optional, Literal, Any
import os, json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src.utils.misc import column_to_label
from src.utils.logging import *
from src.database import *


def __get_raw_and_aggregate_metric_name(metric: str) -> tuple[str, str]:
    match metric:
        case 'R2':
            raw_name, aggregated_name = 'R2Score', 'R2'
        case 'RD' | 'RelativeDistance':
            raw_name, aggregated_name = 'RelativeDistance', 'RelativeDistance'
        case _:
            raise ValueError(f"Unknown or not-implemented metric '{metric}'")
    return raw_name, aggregated_name


def mean_plots(
    db: SecureMLExperimentDB, savepath: Optional[str], filename: str = 'mean_plots.json',
    metric: str = 'R2', set_type: Literal['eval', 'test'] = 'test', *, show: bool = True,
    legend_fontsize: int = 12, xlabel_size: int = 12, ylabel_size: int = 12,
    ticks_fontsize: int = 10, y_range: np.ndarray = np.arange(0.70, 0.94, 0.04)
):
    with open(filename, 'r') as fp:
        config = json.load(fp)
    plot_data = {}
    num_campaigns = None
    exp_query_dict = {}
    for field, model_class in zip(
        ['general', 'scenario', 'architecture', 'loss', 'optimizer', 'scheduler', 'early_stopping', 'strategy'],
        [General, Scenario, Architecture, Loss, Optimizer, Scheduler, EarlyStopping, Strategy]
    ):
        query_dict = config[field]
        record = db.get_first(model_class, query_dict, )
        assert record is not None, \
            f"Check your config file for '{field}', as the corresponding query dict '{query_dict}' matches no records in the database"
        record_id = record['id']
        exp_query_dict[f"id_{field}"] = record_id
        if field == 'general':
            num_campaigns = record['num_campaigns']
    # Impose NO Active Learning
    exp_query_dict['id_active_learning'] = None
    exp_dict = db.get_first(Experiment, exp_query_dict)
    x_values = np.arange(1, num_campaigns + 1)
    assert exp_dict is not None, f"Check your experiment query conditions: '{exp_query_dict}', as there were no eligible records in the database"
    raw_name, aggregated_name = __get_raw_and_aggregate_metric_name(metric)
    raw_logs = np.array(exp_dict['logs']['raw_metrics'][f"{set_type.capitalize()}_{raw_name}"]['mean'], dtype=np.float32)
    aggregated_logs = np.array(exp_dict['logs']['aggregated_metrics'][f"{set_type.capitalize()}_{aggregated_name}"]['mean'], dtype=np.float32)
    assert len(raw_logs) == num_campaigns**2
    assert len(aggregated_logs) == num_campaigns
    # First plot data for each eval experience
    for j in range(num_campaigns):
        eval_exp_values = raw_logs[range(j, num_campaigns**2, num_campaigns)]
        plt.plot(x_values, eval_exp_values, label=f"Experimental Campaign {j + 1}", marker='o', linestyle='-')
    # Then plot overall mean
    plt.plot(x_values, aggregated_logs, label=f"Overall (Weighted) Mean", color='black', marker='o', linestyle='-')
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
    parser.add_argument('--config', type=str, default='mean_plots.json')
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

    args = parser.parse_args()
    mean_plots(
        db, args.savepath, args.config, args.metric, args.set_type, show=True, legend_fontsize=args.legend_size,
        xlabel_size=args.xlabel_size, ylabel_size=args.ylabel_size, ticks_fontsize=args.ticks_size,
        y_range=np.arange(args.y_start, args.y_stop, args.y_step)
    )
