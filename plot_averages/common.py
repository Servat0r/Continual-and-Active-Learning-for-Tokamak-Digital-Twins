import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from src.utils import *
from src.ex_post_tests import *

from argparse import ArgumentParser


def get_replay_percentages(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str,
):
    if pow_type == 'lowpow':
        return {5: 500, 10: 2500}
    if simulator_type == 'qualikiz':
        return {5: 2000, 10: 10000}
    elif simulator_type == 'tglf':
        if pow_type == 'mixed':
            return {5: 3000, 10: 12000}
        elif pow_type == 'highpow':
            return {5: 600, 10: 3000}
    else:
        raise ValueError(f"Unknown simulator_type = {simulator_type}")


_base_cl_strategies_dictionary = {
    'Naive': [
        ('Naive', 'Base'),
        #('Naive (dropout = 0.0)', 'Base (p = 0.0)'),
        #('Naive (dropout = 0.25)', 'Base (p = 0.25)'),
        #('Naive (dropout = 0.5)', 'Base'),
        #('Naive (dropout = 0.25)', 'Base'),
        #('Naive (dropout = 0.5)', 'Base (p = 0.5)'),
        #('Naive (dropout = 0.75)', 'Base (p = 0.75)')
    ],
    'Cumulative': [('Cumulative', 'Base')],
    'FromScratchTraining': [('From Scratch', 'Base')],
    'Replay': [
        ('Replay (500)', 'Buffer 500'), ('Replay (600)', 'Buffer 600'),
        ('Replay (1000)', 'Buffer 1000'), ('Replay (2000)', 'Buffer 2000'),
        ('Replay (2500)', 'Buffer 2500'), ('Replay (3000)', 'Buffer 3000'),
        ('Replay (5000)', 'Buffer 5000'),
        ('Replay (10000)', 'Buffer 10000'), ('Replay (12000)', 'Buffer 12000'),
        ('Replay (20000)', 'Buffer 20000'), ('Replay (40000)', 'Buffer 40000')
    ],
    'EWC': [
        ('EWC (0.1)', 'Lambda 0.1'),
        ('EWC (1)', 'Lambda 1'),
        ('EWC (10)', 'Lambda 10')
    ],
    'EWCReplay': [
        ('EWC Replay (1, 1000)', 'Lambda 1 Buffer 1000'),
        ('EWC Replay (1, 2000)', 'Lambda 1 Buffer 2000'),
        ('EWC Replay (0.1, 2000)', 'Lambda 0.1 Buffer 2000'),
        ('EWC Replay (1, 3000)', 'Lambda 1 Buffer 3000'),
        ('EWC Replay (1, 10000)', 'Lambda 1 Buffer 10000'),
        ('EWC Replay (10, 1000)', 'Lambda 10 Buffer 1000'),
        ('EWC Replay (10, 3000)', 'Lambda 10 Buffer 3000'),
        ('EWC Replay (10, 10000)', 'Lambda 10 Buffer 10000'),
        ('EWC Replay (10, 12000)', 'Lambda 10 Buffer 12000')
    ],
    'MAS': [
        ('MAS (0.5, 0.0)', 'Lambda 0.5 Alpha 0.0'),
        ('MAS (1, 0.0)', 'Lambda 1 Alpha 0.0'),
        ('MAS (10, 0.0)', 'Lambda 10 Alpha 0.0')
    ],
    'MASReplay': [
        ('MAS Replay (1, 0.0, 2000)', 'Lambda 1 Alpha 0.0 Buffer 2000'),
        ('MAS Replay (0.1, 0.0, 2500)', 'Lambda 0.1 Alpha 0.0 Buffer 2500'),
        ('MAS Replay (1, 0.0, 3000)', 'Lambda 1 Alpha 0.0 Buffer 3000'),
        ('MAS Replay (1, 0.0, 10000)', 'Lambda 1 Alpha 0.0 Buffer 10000')
    ],
    'GEM': [
        ('GEM (100 per exp)', 'Patterns 100'),
        ('GEM (200 per exp)', 'Patterns 200'),
        ('GEM (300 per exp)', 'Patterns 300'),
        ('GEM (400 per exp)', 'Patterns 400'),
        ('GEM (500 per exp)', 'Patterns 500'),
        ('GEM (1000 per exp)', 'Patterns 1000'),
        ('GEM (1024 per exp)', 'Patterns 1024'),
        ('GEM (1200 per exp)', 'Patterns 1200'),
        ('GEM (2000 per exp)', 'Patterns 2000')
    ],
    'GEMReplay': [
        ('GEM Replay (500 per exp, 2500 Buffer)', 'Patterns 500 Buffer 2500'),
        ('GEM Replay (400 per exp, 2000 Buffer)', 'Patterns 400 Buffer 2000'),
        ('GEM Replay (400 per exp, 10000 Buffer)', 'Patterns 400 Buffer 10000'),
        ('GEM Replay (1000 per exp, 10000 Buffer)', 'Patterns 1000 Buffer 10000'),
        ('GEM Replay (300 per exp, 3000 Buffer)', 'Patterns 300 Buffer 3000')
    ],
    'VariableGEM': [
        ('VariableGEM (2000, 2000, 400)', 'Patterns 2000 2000 400')
    ],
    'ConstantSizeGEM': [
        ('Constant-Size GEM (2000)', 'Memory 2000'),
        ('Constant-Size GEM (3000)', 'Memory 3000'),
        ('Constant-Size GEM (10000)', 'Memory 10000')
    ],
    'LFL': [
        ('LFL (0.1)', 'Lambda 0.1'),
        ('LFL (0.25)', 'Lambda 0.25'),
        ('LFL (0.5)', 'Lambda 0.5'),
        ('LFL (1)', 'Lambda 1'),
        ('LFL (2)', 'Lambda 2'),
        ('LFL (5)', 'Lambda 5')#,
        #('LFL (10)', 'Lambda 10')
    ],
    'DoubleLFL': [
        ('DoubleLFL (1)', 'Lambda 1'),
        ('DoubleLFL (2)', 'Lambda 2'),
        ('DoubleLFL (10)', '(test) Lambda 10')
    ],
    'LFLEWC': [
        ('LFLEWC (LFL = 1, EWC = 1)', 'Lambda 1 1'),
        ('LFLEWC (LFL = 2, EWC = 2)', 'Lambda 2 2'),
        ('LFLEWC (LFL = 2, EWC = 10)', 'Lambda 2 10')
    ],
    'LFLReplay': [
        ('LFL Replay (0.5, 1000 Buffer)', 'Lambda 0.5 Buffer 1000'),
        ('LFL Replay (0.5, 2000 Buffer)', 'Lambda 0.5 Buffer 2000'),
        ('LFL Replay (1, 2000 Buffer)', 'Lambda 1 Buffer 2000')
    ],
    'SI': [
        ('SI (0.1)', 'Lambda 0.1'),
        ('SI (1)', 'Lambda 1')
    ]
}

def get_cl_strategies_dictionary(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str,
):
    results = get_datasets_sizes_report(simulator_type, pow_type, cluster_type, dataset_type, task, verbose=False)
    percentages = get_replay_percentages(simulator_type, pow_type, cluster_type, dataset_type, task)
    final_train_size = results['final']['train']
    perc_1, perc_5, perc_10 = int(0.01 * final_train_size), int(0.05 * final_train_size), int(0.1 * final_train_size)
    perc_replay = [
        #(f'%Replay 1% (up to {perc_1})', f'Percentage 1% Min {percentages'),
        (f'%Replay 5% (up to {perc_5})', f'Percentage 5% Min {percentages[5]}'),
        (f'%Replay 10% (up to {perc_10})', f'Percentage 10% Min {percentages[10]}')
    ]
    cl_strategies_dictionary = _base_cl_strategies_dictionary.copy()
    cl_strategies_dictionary['PercentageReplay'] = perc_replay
    return cl_strategies_dictionary


# For the "most interesting" strategies ONLY
def _build_entries(
    data: dict[str, list[tuple[str, str]]], gem_sizes: list[int], replay_sizes: list[int], gem_replay_sizes: list[tuple[int, int]],
    ewc_lambda: int = 1, mas_lambda: int = 1, mas_alpha: int = 0
) -> dict[str, list[tuple[str, str]]]:
    data['Replay'] = [(f'Replay ({size})', f'Buffer {size}') for size in replay_sizes]
    data['EWCReplay'] = [(f'EWCReplay ({ewc_lambda}, {size})', f'Lambda {ewc_lambda} Buffer {size}') for size in replay_sizes]
    data['MASReplay'] = [
        (f'MASReplay ({mas_lambda}, {mas_alpha}, {size})', f'Lambda {mas_lambda} Alpha {float(mas_alpha)} Buffer {size}') \
        for size in replay_sizes
    ]
    data['GEM'] = [(f'GEM ({size})', f'Patterns {size}') for size in gem_sizes]
    data['GEMReplay'] = [(f'GEMReplay ({size[0], size[1]})', f'Patterns {size[0]} Buffer {size[1]}') for size in gem_replay_sizes]
    print(data)
    return data


def get_cl_strategies_interest_dictionary(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str,
):
    results = get_datasets_sizes_report(simulator_type, pow_type, cluster_type, dataset_type, task, verbose=False)
    percentages = get_replay_percentages(simulator_type, pow_type, cluster_type, dataset_type, task)
    final_train_size = results['final']['train']
    perc_10 = int(0.1 * final_train_size)
    perc_replay = [
        (f'%Replay 10% (up to {perc_10})', f'Percentage 10% Min {percentages[10]}')
    ]
    base_dict = {
        'Naive': [('Naive', 'Base')],
        'Cumulative': [('Cumulative', 'Base')],
        'LFL': [('LFL (1)', 'Lambda 1')]
    }
    if simulator_type == 'qualikiz':
        if pow_type in ['highpow', 'mixed']:
            _build_entries(
                base_dict, gem_sizes=[1000, 2000], replay_sizes=[2000, 10000], gem_replay_sizes=[(400, 2000), (1000, 10000)]
            )
        else:
            _build_entries(
                base_dict, gem_sizes=[500], replay_sizes=[2500], gem_replay_sizes=[(500, 2500)], ewc_lambda=0.1, mas_lambda=0.1
            )
    elif simulator_type == 'tglf':
        if pow_type == 'highpow':
            _build_entries(
                base_dict, gem_sizes=[400, 1024], replay_sizes=[3000], gem_replay_sizes=[(300, 3000)]
            )
        elif pow_type == 'lowpow':
            _build_entries(
                base_dict, gem_sizes=[400, 1000], replay_sizes=[2500], gem_replay_sizes=[(400, 2000)]
            )
        else:
            _build_entries(
                base_dict, gem_sizes=[1200], replay_sizes=[12000], gem_replay_sizes=[(500, 3000)]
            )
    cl_strategies_dictionary = base_dict #_base_cl_strategies_interest_dictionary.copy()
    cl_strategies_dictionary['PercentageReplay'] = perc_replay
    return cl_strategies_dictionary


__al_cl_strategies = ['Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'EWCReplay', 'GEM']

# AL(CL)
def get_al_cl_strategies_dictionary(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str
):
    cl_strategies_dictionary = get_cl_strategies_dictionary(
        simulator_type, pow_type, cluster_type, dataset_type, task
    )
    al_cl_strategies_dictionary = {
        k: v for k, v in cl_strategies_dictionary.items() if k in __al_cl_strategies
    }
    return al_cl_strategies_dictionary


def get_al_cl_strategies_interest_dictionary(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str
):
    cl_strategies_interest_dictionary = get_cl_strategies_interest_dictionary(
        simulator_type, pow_type, cluster_type, dataset_type, task
    )
    al_cl_strategies_interest_dictionary = {
        k: v for k, v in cl_strategies_interest_dictionary.items() if k in __al_cl_strategies
    }
    return al_cl_strategies_interest_dictionary


al_cl_methods_dictionary = {
    'random_sketch_grad': 'Random (grad kernel)',
    'random_sketch_ll': 'Random (ll kernel)',
    'batchbald': 'BatchBALD',
    'badge': 'Badge',
    'coreset': 'CoreSets',
    'lcmd_sketch_grad': 'LCMD (grad kernel)'
}

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


def get_savepath(
    args, include_future_experiences: bool, plot_metric_name: str,
    is_active_learning: bool, cl_folder_name: str, al_cl_folder_name: str,
    strategies: list[str] = [],
):
    outputs = args.outputs
    simulator_type = args.simulator_type
    pow_type = args.pow_type
    cluster_type = args.cluster_type
    set_type = args.set_type
    include_future_experiences_str = '/Full Experiences' if include_future_experiences else ''
    conf_str = f"{outputs}/{simulator_type}/{pow_type}/{cluster_type}/{plot_metric_name}{include_future_experiences_str}"
    
    if is_active_learning:
        full_first_set_str = ("" if args.full_first_set else "Non-") + "Full First Set"
        reload_weights_str = ("" if args.reload_weights else "No ") + "Reload Weights"
        downsampling_str = f"Downsampling {args.downsampling}"
        conf_str = f"{conf_str}/Batches {args.al_batch_size} {args.al_max_batch_size}/" + \
            f"{full_first_set_str}/{reload_weights_str}/{downsampling_str}"

    set_type_str = "Eval" if set_type == 'eval' else "Test"
    if args.mode == 'cl':
        savefolder = f"plots/Pure CL/{cl_folder_name}/{conf_str}/{set_type_str}"
    else:
        savefolder = f"plots/AL(CL)/{al_cl_folder_name}/{conf_str}/{set_type_str}"
    savepath = f"{savefolder}/{', '.join(strategies)} {args.hidden_size}-{args.hidden_layers} on {set_type_str} Set.png"
    return savefolder, savepath

def common_parser_build(
    with_mode: bool = True, with_active_learning: bool = True,
    with_al_method: bool = True, with_strategies: bool = True,
    with_other_params: bool = True
):
    parser = ArgumentParser()
    if with_mode:
        parser.add_argument('--mode', type=str, default='cl')
    parser.add_argument('--exclude_naive', type=bool, default=False,
                       help='Exclude Naive strategy from plots')
    parser.add_argument('--exclude_cumulative', type=bool, default=False,
                       help='Exclude Cumulative strategy from plots')
    if with_other_params:
        parser.add_argument('--batch_size', type=int, default=4096)
        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--internal_metric_name', type=str, default='R2Score_Exp')
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--count', type=int, default=-1)
    parser.add_argument('--outputs', type=str, default='efe_efi_pfe_pfi')
    parser.add_argument('--show', type=int, default=1)
    parser.add_argument('--include_std', type=int, default=1)
    parser.add_argument('--savepath', type=str, default=None)
    if with_strategies:
        parser.add_argument('--strategies', nargs='+', type=str, default=[],
                        help='CL Strategies to use')
    if with_active_learning:
        if with_al_method:
            parser.add_argument('--al_method', type=str, default='random_sketch_grad',
                                help='Single AL method to use (only for al_cl mode)')
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
    parser.add_argument('--include_future_experiences', type=bool, default=False,
                        help='Whether to include or not future experiences in mean/std calculation (for Forward Transfer)')
    parser.add_argument('--set_type', type=str, default='eval', help='Set type: either "eval" or "test"')
    return parser
