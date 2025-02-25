import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from src.utils import *
from src.ex_post_tests import *

from argparse import ArgumentParser


# CL (TODO QUALIKIZ ONLY!)
_base_cl_strategies_dictionary = {
    'Naive': [('Naive', 'Base')],
    'Cumulative': [('Cumulative', 'Base')],
    'FromScratchTraining': [('From Scratch', 'Base')],
    'Replay': [
        ('Replay (600)', 'Buffer 600'), ('Replay (1000)', 'Buffer 1000'),
        ('Replay (2000)', 'Buffer 2000'), ('Replay (3000)', 'Buffer 3000'),
        ('Replay (10000)', 'Buffer 10000')
    ],
    #'PercentageReplay': [
    #    ('%Replay 1% (200 => 2299)', 'Percentage 1%'),
    #    ('%Replay 5% (2000 => 11495)', 'Percentage 5%'),
    #    ('%Replay 10% (10000 => 22990)', 'Percentage 10%')
    #],
    'EWC': [
        ('EWC (0.1)', 'Lambda 0.1'),
        ('EWC (1)', 'Lambda 1'),
        ('EWC (10)', 'Lambda 10')
    ],
    'EWCReplay': [
        ('EWC Replay (1, 1000)', 'Lambda 1 Buffer 1000'),
        ('EWC Replay (1, 2000)', 'Lambda 1 Buffer 2000'),
        ('EWC Replay (1, 3000)', 'Lambda 1 Buffer 3000'),
        ('EWC Replay (1, 10000)', 'Lambda 1 Buffer 10000'),
        ('EWC Replay (10, 1000)', 'Lambda 10 Buffer 1000'),
        ('EWC Replay (10, 3000)', 'Lambda 10 Buffer 3000'),
        ('EWC Replay (10, 10000)', 'Lambda 10 Buffer 10000')
    ],
    'EWCMAS': [
        ('EWCMAS (1, 1, 0.0)', 'EWC 1 MAS 1 Alpha 0.0')
    ],
    'MAS': [
        ('MAS (0.5, 0.0)', 'Lambda 0.5 Alpha 0.0'),
        ('MAS (1, 0.0)', 'Lambda 1 Alpha 0.0'),
        ('MAS (10, 0.0)', 'Lambda 10 Alpha 0.0')
    ],
    'IncrementalMAS': [
        ('IncrementalMAS (0.25, 0.25, 1.0)', 'Lambda 0.25 0.25 1.0 Alpha 0.0'),
        ('IncrementalMAS (0.5, 0.5, 5.0)', 'Lambda 0.5 0.5 5.0 Alpha 0.0'),
        ('IncrementalMAS (0.0, 1.0, 5.0)', 'Lambda 0.0 1.0 5.0 Alpha 0.0'),
        ('IncrementalMAS (0.5, 1.0, 5.0)', 'Lambda 0.5 1.0 5.0 Alpha 0.0')
    ],
    'MASReplay': [
        ('MAS Replay (1, 0.0, 1000)', 'Lambda 1 Alpha 0.0 Buffer 1000'),
        ('MAS Replay (1, 0.0, 2000)', 'Lambda 1 Alpha 0.0 Buffer 2000'),
        ('MAS Replay (1, 0.0, 10000)', 'Lambda 1 Alpha 0.0 Buffer 10000')
    ],
    'GEM': [
        ('GEM (400 per exp)', 'Patterns 400'),
        ('GEM (1000 per exp)', 'Patterns 1000'),
        ('GEM (2000 per exp)', 'Patterns 2000')
    ],
    'GEMReplay': [
        ('GEM Replay (400 per exp, 1000 buffer)', 'Patterns 400 Buffer 1000'),
        ('GEM Replay (400 per exp, 10000 buffer)', 'Patterns 400 Buffer 10000'),
        ('GEM Replay (1000 per exp, 1000 buffer)', 'Patterns 1000 Buffer 1000'),
    ],
    'VariableGEM': [
        ('VariableGEM (2000, 2000, 400)', 'Patterns 2000 2000 400')
    ],
    'LFL': [
        ('LFL (0.25)', 'Lambda 0.25'),
        ('LFL (0.5)', 'Lambda 0.5'),
        ('LFL (1)', 'Lambda 1'),
        ('LFL (2)', 'Lambda 2'),
        ('LFL (5)', 'Lambda 5')
    ],
    'DoubleLFL': [
        ('DoubleLFL (1)', 'Lambda 1'),
        ('DoubleLFL (2)', 'Lambda 2'),
        ('DoubleLFL (10)', '(test) Lambda 10')
    ],
    'LFLReplay': [
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
    final_train_size = results['final']['train']
    perc_1, perc_5, perc_10 = int(0.01 * final_train_size), int(0.05 * final_train_size), int(0.1 * final_train_size)
    perc_replay = [
        (f'%Replay 1% (up to {perc_1})', 'Percentage 1%'),
        (f'%Replay 5% (up to {perc_5})', 'Percentage 5%'),
        (f'%Replay 10% (up to {perc_10})', 'Percentage 10%')
    ]
    cl_strategies_dictionary = _base_cl_strategies_dictionary.copy()
    cl_strategies_dictionary['PercentageReplay'] = perc_replay
    return cl_strategies_dictionary


# This is intended to be the dictionary of "most interesting" configurations for each strategy
_base_cl_strategies_interest_dictionary = {
    'Naive': [('Naive', 'Base')],
    'Cumulative': [('Cumulative', 'Base')],
    'Replay': [
        ('Replay (2000)', 'Buffer 2000'), ('Replay (10000)', 'Buffer 10000')
    ],
    'PercentageReplay': [
        ('%Replay 10% (10000 => 22990)', 'Percentage 10%')
    ],
    'EWC': [
        ('EWC (10)', 'Lambda 10')
    ],
    'EWCReplay': [
        ('EWC Replay (1, 10000)', 'Lambda 1 Buffer 10000')
    ],
    'MAS': [
        ('MAS (1, 0.0)', 'Lambda 1 Alpha 0.0'),
    ],
    'MASReplay': [
        ('MAS Replay (1, 0.0, 10000)', 'Lambda 1 Alpha 0.0 Buffer 10000')
    ],
    'GEM': [
        ('GEM (1000)', 'Patterns 1000'),
        ('GEM (2000)', 'Patterns 2000')
    ],
    'GEMReplay': [
        ('GEM Replay (400, 10000)', 'Patterns 400 Buffer 10000'),
        ('GEM Replay (1000, 1000)', 'Patterns 1000 Buffer 1000'),
    ],
    'LFL': [
        ('LFL (2)', 'Lambda 2')
    ],
}

def get_cl_strategies_interest_dictionary(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str, task: str,
):
    results = get_datasets_sizes_report(simulator_type, pow_type, cluster_type, dataset_type, task, verbose=False)
    final_train_size = results['final']['train']
    perc_10 = int(0.1 * final_train_size)
    perc_replay = [
        (f'%Replay 10% (up to {perc_10})', 'Percentage 10%')
    ]
    cl_strategies_dictionary = _base_cl_strategies_interest_dictionary.copy()
    cl_strategies_dictionary['PercentageReplay'] = perc_replay
    return cl_strategies_dictionary


__al_cl_strategies = ['Naive', 'Cumulative', 'Replay', 'PercentageReplay', 'EWCReplay']

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


def common_parser_build(
    with_mode: bool = True, with_active_learning: bool = True,
    with_al_method: bool = True, with_strategies: bool = True
):
    parser = ArgumentParser()
    if with_mode:
        parser.add_argument('--mode', type=str, default='cl')
    parser.add_argument('--exclude_naive', type=bool, default=False,
                       help='Exclude Naive strategy from plots')
    parser.add_argument('--exclude_cumulative', type=bool, default=False,
                       help='Exclude Cumulative strategy from plots')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--internal_metric_name', type=str, default='R2Score_Exp')
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--count', type=int, default=-1)
    parser.add_argument('--outputs', type=str, default='efe_efi_pfe_pfi')
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
    return parser
