# NOTE: QuaLiKiz only!
import json, sys
import numpy as np
import matplotlib.pyplot as plt

from src.utils.scenarios import *
from src.utils.plots import DEFAULT_COLORS, DEFAULT_LINESTYLES
from src.utils.datasets import QUALIKIZ_HIGHPOW_OUTPUTS
from src.ex_post_tests import get_db_filename
from argparse import ArgumentParser


cl_hidden_size = 1024
al_cl_hidden_size = 256
hidden_layers = 2


def get_label_name(item: dict):
    strategy_name, plot_name = item['strategy'], item['plot_name']
    if len(plot_name) > 0:
        return f"{strategy_name} ({plot_name})"
    else:
        return strategy_name


def standard_method_to_label(method: str):
    if method == 'random_sketch_grad':
        return "Random (grad kernel)"
    elif method == 'batchbald':
        return "BatchBALD"
    elif method == 'badge':
        return "BADGE"
    elif method == 'lcmd_sketch_grad':
        return "LCMD"
    else:
        return method.replace('_', ' ')


def column_to_label(column: str):
    if column in ("R2", "R2Score_Exp"):
        return r"$R^2$"
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


valid_strategies = {'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'GEM', 'EWCReplay', 'GEMReplay'}


def plot_list(filepath: str, column: str, savepath: str = None):
    db = json.load(open(filepath, 'r'))
    index = 0
    for item in db:
        if (item['mode'].upper() == 'CL') and (item['strategy'] in valid_strategies) and (item['hidden_size'] == cl_hidden_size):
            color, linestyle = DEFAULT_COLORS[index], DEFAULT_LINESTYLES[index]
            label_name = get_label_name(item)
            values: list = item[column]
            xs: np.ndarray = np.arange(len(values))
            start_index = 1 if column == 'R' else 0
            plt.plot(
                xs[start_index:], values[start_index:], label=label_name,
                marker='o', color=color, linestyle=linestyle
            )
            index += 1
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.xlabel(r"Experimental Campaign ($i$)")
    plt.ylabel(column_to_label(column))
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_al_cl_single_strategy(
    config: ScenarioConfig, al_config: ActiveLearningConfig,
    strategy: str, extra_log_folder: str, filepath: str,
    column: str, savepath: str = None
):
    db = json.load(open(filepath, 'r'))
    al_fields = ["batch_size", "max_batch_size", "full_first_set", "first_set_size", "downsampling_factor", "hidden_size"]
    filter_data = {
        **config.to_dict(),
        "mode": "AL(CL)",
        "strategy": strategy,
        "extra_log_folder": extra_log_folder,
        "hidden_size": al_cl_hidden_size,
        **dict(filter(lambda x: x[0] in al_fields, al_config.to_dict().items()))
    }
    baseline_filter_data = {
        **config.to_dict(),
        "mode": "CL",
        "strategy": strategy,
        "extra_log_folder": extra_log_folder,
        #"hidden_size": al_cl_hidden_size,
    }
    naive_cl_filter_data = {
        **config.to_dict(),
        "mode": "CL",
        "strategy": "Naive",
        "extra_log_folder": "Base",
        "hidden_size": 256
    }
    print(filter_data, baseline_filter_data, sep='\n')
    index = 0
    for item in db:
        if all(item.get(field) == value for field, value in filter_data.items()) or \
            all(item.get(field) == value for field, value in baseline_filter_data.items()) or \
            all(item.get(field) == value for field, value in naive_cl_filter_data.items()):
            color, linestyle = DEFAULT_COLORS[index], DEFAULT_LINESTYLES[index]
            if item['mode'].upper() == 'AL(CL)':
                label_name = standard_method_to_label(item['standard_method'])
            elif item['strategy'] == 'Naive' and item['hidden_size'] == al_cl_hidden_size:
                label_name = f"Naive CL Baseline ({al_cl_hidden_size} hidden size)"
            elif item['hidden_size'] == al_cl_hidden_size:
                label_name = f"{strategy} CL Baseline ({al_cl_hidden_size} hidden size)"
            elif item['hidden_size'] == cl_hidden_size:
                label_name = f"{strategy} CL Baseline ({cl_hidden_size} hidden size)"
            values: list = item[column]
            xs: np.ndarray = np.arange(len(values))
            start_index = 1 if column == 'R' else 0
            plt.plot(
                xs[start_index:], values[start_index:], label=label_name,
                marker='o', color=color, linestyle=linestyle
            )
            index += 1
    print(index)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.xlabel(r"Experimental Campaign ($i$)")
    plt.ylabel(column_to_label(column))
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='CL')
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='mixed')
    parser.add_argument('--cluster_type', type=str, default='beta_based')
    parser.add_argument('--metric', type=str, default='R2',
                      help='Metric to plot (R2, R, times, cumulative_times, time_ratios, or num_epochs)')
    parser.add_argument('--strategy', type=str, default='Naive')
    parser.add_argument('--extra_log_folder', type=str, default='Base')
    args = parser.parse_args()

    config = ScenarioConfig(
        simulator_type=args.simulator_type,
        pow_type=args.pow_type,
        cluster_type=args.cluster_type,
        dataset_type='not_null',
        task='regression',
        outputs=QUALIKIZ_HIGHPOW_OUTPUTS
    )

    if args.mode.upper() == 'CL':
        metric = args.metric
        sim_type = 'qlk' if config.simulator_type == 'qualikiz' else config.simulator_type
        savepath = f"plots/{args.mode}_{metric}_{sim_type}_{config.pow_type}_{config.cluster_type[:-6]}.pdf"
        filepath = get_db_filename(config)
        plot_list(filepath, metric, savepath)
    else:
        metric = args.metric
        sim_type = 'qlk' if config.simulator_type == 'qualikiz' else config.simulator_type
        savepath = f"plots/{args.mode}_{metric}_{sim_type}_{config.pow_type}_{config.cluster_type[:-6]}_{args.strategy}_{args.extra_log_folder}.pdf"
        filepath = get_db_filename(config)
        al_config = ActiveLearningConfig(
            framework='bmdal',
            batch_size=256,
            max_batch_size=1024,
            reload_initial_weights=False,
            standard_method='random_sketch_grad',
            full_first_set=True,
            first_set_size=5120,
            downsampling_factor=0.5
        )
        plot_al_cl_single_strategy(
            config, al_config, args.strategy, args.extra_log_folder, filepath, metric, savepath
        )
