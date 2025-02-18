import os
import json
from argparse import ArgumentParser
from rich import print

from src.utils import *
from src.ex_post_tests import *


def _make_str_list(prefix: str = '', data: list = [], suffix: str = ''):
    return [prefix + item + suffix for item in data]


# CL
cl_strategies = [
    'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'EWC',
    'EWCReplay', 'MAS', 'MASReplay', 'GEM', 'GEMReplay', 'LFL', 'SI'
]

cl_extra_log_folders = [
    ['Base'], ['Base'], ['Buffer 500', 'Buffer 1000', 'Buffer 2000', 'Buffer 10000'],
    ['Percentage 1%', 'Percentage 5%', 'Percentage 10%'], ['Lambda 1', 'Lambda 10'],
    ['Lambda 10 Buffer 1000'], ['Lambda 1 Alpha 0.0', 'Lambda 10 Alpha 0.0'],
    ['Lambda 1 Alpha 0.0 Buffer 1000', 'Lambda 10 Alpha 0,5 Buffer 1000'],
    ['Patterns 100', 'Patterns 400', 'Patterns 1000', 'Patterns 1000 Gamma 0.1', 'Patterns 1000 Gamma 0.25', 'Patterns 2000'],
    ['Patterns 100 Buffer 1000'], ['Lambda 1', 'Lambda 10'], ['Lambda 0.1', 'Lambda 1']
]

cl_proxy_names = [
    ['Naive'], ['Cumulative'],
    _make_str_list('Replay (', ['500', '1000', '2000', '10000'], ')'),
    _make_str_list('Percentage Replay (', ['1', '5', '10'], '%)'),
    ['EWC (1)', 'EWC (10)'], ['EWCReplay (10, 1000)'],
    ['MAS (1, 0.0)', 'MAS (10, 0.0)'],
    ['MAS Replay (1, 0.0, 1000)', 'MAS Replay (10, 0.5, 1000)'],
    ['GEM (100 per exp)', 'GEM (400 per exp)', 'GEM (1000 per exp)', 'GEM (1000 per exp, gamma = 0.1)', 'GEM (1000 per exp, gamma = 0.25)', 'GEM (2000 per exp)'],
    ['GEMReplay (100, 1000)'],
    ['LFL (1)', 'LFL (10)'], ['SI (0.1)', 'SI (1)']
]

# AL(CL)
al_cl_strategies = [
    'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'GEM'
]

al_cl_extra_log_folders = [
    ['Base'], ['Base'], ['Buffer 1000'], ['Percentage 10% Base 10000'], ['Patterns 1000']
]

al_cl_proxy_names = [
    ['Naive'], ['Cumulative'], ['Replay (1000)'], ['Percentage Replay (10%, 10000)'], ['GEM (1000 per exp)']
]

al_cl_methods = [
    'random_sketch_grad', 'random_sketch_ll', 'batchbald', 'badge', 'coreset', 'lcmd_sketch_grad'
]

al_cl_methods_proxies = [
    'Random (grad kernel)', 'Random (ll kernel)', 'BatchBALD', 'Badge', 'CoreSets', 'LCMD (grad kernel)'
]

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


if __name__ == '__main__':
    parser = ArgumentParser()
    # Modes:
    # cl: default CL strategies, compares multiple CL strategies together
    # al_cl_ms: AL(CL), choose one CL strategy and compares multiple AL methods over it
    # al_cl_sm: AL(CL), choose one AL method and compares multiple CL strategies over it
    parser.add_argument('--mode', type=str, default='cl')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--internal_metric_name', type=str, default='Forgetting_Exp')
    parser.add_argument('--output_prefix', type=str, default='forgetting')
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--count', type=int, default=-1)
    parser.add_argument(
        '--strategy_mask', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        help='Mask of 0s and 1s to filter strategies (e.g. 1 0 1 1 0 1 1 1 1 0 0)'
    )
    parser.add_argument('--al_method', type=str, default=None,
                        help='Single AL method to use (only for al_cl_sm modes)')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Single CL strategy to use (only for al_cl_ms modes)')
    parser.add_argument(
        '--al_methods_mask', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1],
        help='Mask of 0s and 1s to filter AL methods (random_sketch_grad, random_sketch_ll, ' + \
            'batchbald, badge, coreset, lcmd_sketch_grad)'
    )
    parser.add_argument('--al_batch_size', type=int, default=128,
                       help='Batch size for active learning')
    parser.add_argument('--al_max_batch_size', type=int, default=2048,
                       help='Maximum batch size for active learning')
    parser.add_argument('--full_first_set', type=bool, default=False,
                       help='Whether to use full first set in active learning')
    parser.add_argument('--reload_weights', type=bool, default=False,
                       help='Whether to reload weights between active learning iterations')
    parser.add_argument('--downsampling', type=float, default=0.5,
                       help='Downsampling ratio for active learning')
    args = parser.parse_args()
    is_active_learning = args.mode in ['al_cl_sm', 'al_cl_ms']

    if args.mode == 'cl':
        strategies = cl_strategies
        extra_log_folders = cl_extra_log_folders
        proxies = cl_proxy_names
    elif is_active_learning:
        strategies = al_cl_strategies
        extra_log_folders = al_cl_extra_log_folders
        proxies = al_cl_proxy_names
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if is_active_learning and (len(args.strategy_mask) > len(strategies)):
        args.strategy_mask = args.strategy_mask[:len(strategies)]
    # Filter strategies, extra_log_folders and proxies based on strategy_mask
    if len(args.strategy_mask) != len(strategies):
        raise ValueError(f"Strategy mask length ({len(args.strategy_mask)}) must match number of strategies ({len(strategies)})")
    
    if args.mode in ['cl', 'al_cl_sm']: # Pure CL, Single method Multiple strategies
        strategies = [s for s, m in zip(strategies, args.strategy_mask) if m]
        extra_log_folders = [f for f, m in zip(extra_log_folders, args.strategy_mask) if m]
        proxies = [p for p, m in zip(proxies, args.strategy_mask) if m]
        if args.mode == 'cl':
            al_methods = []
            al_proxies = []
        else:
            for m, p in zip(al_cl_methods, al_cl_methods_proxies):
                if m == args.al_method:
                    al_methods = [m]
                    al_proxies = [p]
                    break
    elif args.mode == 'al_cl_ms': # Multiple methods Single strategy
        single_strategy = args.strategy
        for s, f, p in zip(strategies, extra_log_folders, proxies):
            if s == single_strategy:
                strategies = [s]
                extra_log_folders = f
                proxies = p
                break
        al_methods = [a for a, m in zip(al_cl_methods, args.al_methods_mask) if m]
        al_proxies = [p for p, m in zip(al_cl_methods_proxies, args.al_methods_mask) if m]
    
    outputs = 'efe_efi_pfe_pfi'

    simulator_prefix = simulator_prefixes[args.simulator_type]
    if args.internal_metric_name.endswith('_Exp'):
        plot_metric_name = args.internal_metric_name[:-4]
    else:
        plot_metric_name = args.internal_metric_name[:]
    
    if args.mode in ['cl', 'al_cl_sm']:
        tqdm_size = sum([len(folder) for folder in extra_log_folders])
        strategies_dict = {}
        colors_dict = {}
        
        index = 0
        pow_type, cluster_type, task, dataset_type = args.pow_type, args.cluster_type, args.task, args.dataset_type
        for (strategy, proxy_list, folders) in zip(strategies, proxies, extra_log_folders):
            for proxy, folder in zip(proxy_list, folders):
                extra_log_folder = simulator_prefix + folder
                try:
                    folder_path = get_log_folder(
                        pow_type, cluster_type, task, dataset_type, outputs, strategy, args.hidden_size,
                        args.hidden_layers, args.batch_size, is_active_learning, args.al_batch_size,
                        args.al_max_batch_size, args.al_method, args.full_first_set, args.reload_weights,
                        args.downsampling, extra_log_folder, count=-1, simulator_type=args.simulator_type
                    )
                    print(f"[red]Strategy: {strategy}, Proxy: {proxy}, Folder: {extra_log_folder}[/red]")
                    strategies_dict[proxy] = (strategy, extra_log_folder)
                    colors_dict[proxy] = (colors[index], linestyles[index])
                    index += 1
                except FileNotFoundError as ex:
                    print(f"[purple]{ex.args}[/purple]")

        if args.mode == 'cl':
            savefolder = f"plots/Pure CL"
        else:
            savefolder = f"plots/AL(CL) Strategy Comparisons"
        savefolder = f"{savefolder}/{plot_metric_name}/{pow_type}/{cluster_type}/{dataset_type}/{task}/"
        savepath = f"{savefolder}/{', '.join(strategies)} {args.hidden_size}-{args.hidden_layers} on Eval Set.png"
        os.makedirs(savefolder, exist_ok=True)

        mean_std_strategy_plots_wrapper(
            pow_type, cluster_type, dataset_type, task, outputs, strategies_dict,
            internal_metric_name=args.internal_metric_name, plot_metric_name=plot_metric_name,
            simulator_type=args.simulator_type, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
            batch_size=args.batch_size, active_learning=is_active_learning, # TODO Completare!
            show=True, save=True, savepath=savepath, grid=True, legend=True, count=-1,
            colors_and_linestyle_dict=colors_dict
        )
    else:
        # AL(CL) method comparisons
        tqdm_size = len(al_cl_methods)
        al_methods_dict = {}
        colors_dict = {}

        index = 0
        pow_type, cluster_type, task, dataset_type = args.pow_type, args.cluster_type, args.task, args.dataset_type
        strategy = strategies[0]

        for method, proxy in zip(al_cl_methods, al_cl_methods_proxies):
            for extra_log_folder, strategy_name_proxy in zip(extra_log_folders, proxies):
                try:
                    folder_path = get_log_folder(
                        pow_type, cluster_type, task, dataset_type, outputs, strategy, args.hidden_size,
                        args.hidden_layers, args.batch_size, True, args.al_batch_size,
                        args.al_max_batch_size, method, args.full_first_set, args.reload_weights,
                        args.downsampling, extra_log_folder, count=-1, simulator_type=args.simulator_type
                    )
                    print(f"[red]Strategy: {strategy}, AL Method: {method}, Proxy: {proxy}[/red]")
                    final_proxy_name = f"{strategy_name_proxy} - {proxy}"
                    al_methods_dict[final_proxy_name] = (method, extra_log_folder)
                    colors_dict[final_proxy_name] = (colors[index], linestyles[index])
                    index += 1
                except FileNotFoundError as ex:
                    print(f"[purple]{ex.args}[/purple]")

        savefolder = f"plots/AL(CL) Method Comparisons/{plot_metric_name}/{pow_type}/{cluster_type}/{dataset_type}/{task}/{strategy}"
        savepath = f"{savefolder}/AL Methods {args.hidden_size}-{args.hidden_layers} on Eval Set.png"
        os.makedirs(savefolder, exist_ok=True)

        mean_std_al_plots_wrapper(
            pow_type, cluster_type, dataset_type, task, outputs, strategy, al_methods_dict,
            batch_size=args.batch_size, al_batch_size=args.al_batch_size,
            al_max_batch_size=args.al_max_batch_size, 
            full_first_set=args.full_first_set, reload_weights=args.reload_weights,
            downsampling_factor=args.downsampling, internal_metric_name=args.internal_metric_name,
            plot_metric_name=plot_metric_name, simulator_type=args.simulator_type,
            hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
            count=-1, show=True, save=True, savepath=savepath, grid=True, legend=True,
            colors_and_linestyle_dict=colors_dict
        )
