# NOTE: QuaLiKiz only!
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt


simulator_type = "qualikiz"
pow_type = "mixed"
cluster_type = "beta_based" #"wmhd_based" #
dataset_type = "not_null"
task = "regression"


# Colors and linestyles
colors = 2 * [
    # Default matplotlib colors
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    'black', 'gold', 'brown', 'darkblue'
]

linestyles = ['-' for _ in range(len(colors)//2)] + ['--' for _ in range(len(colors)//2)]


def get_label_name(item: dict):
    strategy_name, plot_name = item['strategy'], item['plot_name']
    if len(plot_name) > 0:
        return f"{strategy_name} ({plot_name})"
    else:
        return strategy_name


#valid_strategies = {'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'EWC', 'MAS', 'GEM', 'LFL', 'EWCReplay', 'MASReplay', 'GEMReplay'}
#valid_strategies = {'Naive', 'Cumulative', 'Replay', 'EWC', 'MAS', 'LFL', 'MASReplay'}
valid_strategies = {'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'GEM', 'EWCReplay', 'GEMReplay'}
#valid_strategies = {'Naive', 'Cumulative', 'Replay', 'GEM', 'GEMReplay'}


def plot_list(filepath: str, column: str, savepath: str = None):
    db = json.load(open(filepath, 'r'))
    index = 0
    for item in db:
        if item['strategy'] in valid_strategies:
            color, linestyle = colors[index], linestyles[index]
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
    plt.legend(fontsize=8)
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    argv = sys.argv
    metric = argv[1] if len(sys.argv) >= 2 else 'R2'
    savepath = f"plots/{metric}_qlk_{pow_type}_{cluster_type[:-6]}.pdf"
    filepath = f"db_qlk_{pow_type}_{cluster_type}.json"
    plot_list(filepath, metric, savepath)
