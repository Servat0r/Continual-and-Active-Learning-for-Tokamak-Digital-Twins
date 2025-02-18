import os
import json
from argparse import ArgumentParser
from rich import print

from src.utils import *
from src.ex_post_tests import *


def _make_str_list(prefix: str = '', data: list = [], suffix: str = ''):
    return [prefix + item + suffix for item in data]


simulator_prefixes = {
    'qualikiz': [''],
    'tglf': ['TGLF/']
    #'tglf': ['TGLF/', 'TGLF/MSECOS 0.01 100/']
}

# CL
cl_strategies = [
    'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'EWC',
    'EWCReplay', 'MAS', 'MASReplay', 'GEM', 'GEMReplay', 'LFL', 'SI'
]

cl_extra_log_folders = [
    ['Base'], ['Base'], ['Buffer 500', 'Buffer 1000', 'Buffer 2000', 'Buffer 10000'],
    ['Percentage 1%', 'Percentage 5%', 'Percentage 10%'], ['Lambda 0.01', 'Lambda 1', 'Lambda 10'],
    ['Lambda 10 Buffer 1000'], ['Lambda 1 Alpha 0.0', 'Lambda 10 Alpha 0.0'],
    ['Lambda 1 Alpha 0.0 Buffer 1000', 'Lambda 10 Alpha 0,5 Buffer 1000'],
    ['Patterns 100', 'Patterns 400', 'Patterns 1000', 'Patterns 2000'], ['Patterns 100 Buffer 1000'],
    ['Lambda 1', 'Lambda 10'], ['Lambda 0.1', 'Lambda 1']
]

cl_proxy_names = [
    ['Naive'], ['Cumulative'],
    _make_str_list('Replay (', ['500', '1000', '2000', '10000'], ')'),
    _make_str_list('Percentage Replay (', ['1', '5', '10'], '%)'),
    ['EWC (0.01)', 'EWC (1)', 'EWC (10)'], ['EWCReplay (10, 1000)'],
    ['MAS (1, 0.0)', 'MAS (10, 0.0)'],
    ['MAS Replay (1, 0.0, 1000)', 'MAS Replay (10, 0.5, 1000)'],
    ['GEM (100 per exp)', 'GEM (400 per exp)', 'GEM (1000 per exp)', 'GEM (2000 per exp)'], ['GEMReplay (100, 1000)'],
    ['LFL (1)', 'LFL (10)'], ['SI (0.1)', 'SI (1)']
]
# Extended named colors
extended_colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
    'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown',
    'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
    'cornflowerblue', 'cornsilk', 'crimson', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta',
    'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet',
    'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
    'gold', 'goldenrod', 'gray', 'greenyellow', 'honeydew',
    'hotpink', 'indianred', 'indigo', 'ivory', 'khaki',
    'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
    'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
    'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
    'plum', 'powderblue', 'purple', 'rebeccapurple', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen',
    'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'snow', 'springgreen', 'steelblue', 'tan',
    'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'whitesmoke', 'yellowgreen'
]

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
    'black', 'gold',
]

linestyles = ['-' for _ in range(len(colors)//2)] + ['--' for _ in range(len(colors)//2)]

# AL(CL)
al_cl_strategies = [
    'Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'GEM'
]

al_cl_extra_log_folders = [
    [''], [''], ['Buffer 1000'], ['Percentage 10% Base 10000'], ['Patterns 1000']
]

al_cl_methods = [
    'random_sketch_grad', 'random_sketch_ll', 'batchbald', 'badge', 'coreset', 'lcmd_sketch_grad'
]


if __name__ == '__main__':
    parser = ArgumentParser()
    # Modes:
    # cl: default CL strategies, compares multiple CL strategies together
    # al_cl_sm: AL(CL), choose one CL strategy and compares multiple AL methods over it
    # al_cl_ms: AL(CL), choose one AL method and compares multiple CL strategies over it
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
    parser.add_argument('--strategy', type=str, default=None,
                       help='Single CL strategy to use (only for AL(CL) modes)')
    parser.add_argument(
        '--al_methods_mask', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1],
        help='Mask of 0s and 1s to filter AL methods (random_sketch_grad, random_sketch_ll, batchbald, badge, coreset, lcmd_sketch_grad)'
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

    if args.mode == 'cl':
        strategies = cl_strategies
        extra_log_folders = cl_extra_log_folders
        proxies = cl_proxy_names
    elif args.mode in ['al_cl_sm', 'al_cl_ms']:
        strategies = al_cl_strategies
        extra_log_folders = al_cl_extra_log_folders
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if (args.mode in ['al_cl_sm', 'al_cl_ms']) and (len(args.strategy_mask) > len(strategies)):
        args.strategy_mask = args.strategy_mask[:len(strategies)]
    # Filter strategies, extra_log_folders and proxies based on strategy_mask
    if len(args.strategy_mask) != len(strategies):
        raise ValueError(f"Strategy mask length ({len(args.strategy_mask)}) must match number of strategies ({len(strategies)})")
    
    strategies = [s for s, m in zip(strategies, args.strategy_mask) if m]
    extra_log_folders = [f for f, m in zip(extra_log_folders, args.strategy_mask) if m]
    proxies = [p for p, m in zip(proxies, args.strategy_mask) if m]
    if args.mode in ['al_cl_sm', 'al_cl_ms']:
        ... # TODO: Continuare!
    
    suffix = f'({args.batch_size} batch size) ({args.hidden_size} hidden size)'
    if args.hidden_layers != 2 or args.simulator_type == 'tglf':
        suffix = suffix + f' ({args.hidden_layers} hidden layers)'
    outputs = 'efe_efi_pfe_pfi'

    current_simulator_prefixes = simulator_prefixes[args.simulator_type]
    tqdm_size = len(current_simulator_prefixes) * sum([len(folder) for folder in extra_log_folders])
    strategies_dict = {}
    colors_dict = {}
    
    index = 0
    pow_type, cluster_type, task, dataset_type = args.pow_type, args.cluster_type, args.task, args.dataset_type
    for simulator_prefix in current_simulator_prefixes:
        for (strategy, proxy_list, folders) in zip(strategies, proxies, extra_log_folders):
            for proxy, folder in zip(proxy_list, folders):
                extra_log_folder = simulator_prefix + folder + ' ' + suffix
                folder_path = f'logs/{pow_type}/{cluster_type}/{task}/{dataset_type}/{outputs}/{strategy}/{extra_log_folder}'
                if os.path.exists(folder_path):
                    print(f"[red]Strategy: {strategy}, Proxy: {proxy}, Folder: {extra_log_folder}[/red]")
                    strategies_dict[proxy] = (strategy, extra_log_folder)
                    colors_dict[proxy] = (colors[index], linestyles[index])
                    index += 1

    if args.internal_metric_name.endswith('_Exp'):
        plot_metric_name = args.internal_metric_name[:-4]
    else:
        plot_metric_name = args.internal_metric_name[:]
    savepath = f"plots/Pure CL {pow_type} {cluster_type} " + \
        f"{dataset_type} {task} {plot_metric_name} {strategies} " + \
        f"{args.hidden_size}-{args.hidden_layers} on Eval Set"
    mean_std_strategy_plots_wrapper(
        pow_type, cluster_type, dataset_type, task, outputs, strategies_dict,
        internal_metric_name=args.internal_metric_name, plot_metric_name=plot_metric_name,
        show=True, save=True, savepath=savepath, grid=True, legend=True, count=-1,
        colors_and_linestyle_dict=colors_dict
    )
