import os
from joblib import Parallel, delayed

from src.utils import *
from src.ex_post_tests import *
from argparse import ArgumentParser


single_strategies_list = [
    'Replay PercentageReplay',
    'PercentageReplay',
    'EWCReplay',
    'GEM',
    'GEMReplay',
    'Replay EWCReplay',
    'Replay GEM',
    'Replay LFL',
    'Replay MAS',
    'Replay EWC',
    'Replay SI',
    'GEM GEMReplay',
    'EWC EWCReplay',
    'MAS MASReplay'
]

#multiple_strategies = 'Replay EWCReplay GEM LFL MASReplay GEMReplay'
multiple_strategies = 'Replay EWCReplay GEM GEMReplay'


def task(simulator_type, pow_type, cluster_type, set_type, hidden_size, hidden_layers, item, internal_metric_name):
    print(f"Plotting for {item} ...")
    system_call = \
        f"python -m plot_averages.single_strategy --simulator_type={simulator_type} --pow_type={pow_type} " + \
        f"--cluster_type={cluster_type} --dataset_type=not_null --mode=cl --batch_size=4096 --strategies {item} " + \
        f"--set_type={set_type} --show=0 --hidden_size={hidden_size} --hidden_layers={hidden_layers} " + \
        f"--internal_metric_name={internal_metric_name}"
    if simulator_type == 'tglf':
        system_call = f"{system_call} --exclude_naive=1"
        if pow_type == 'lowpow':
            system_call = f"{system_call} --include_std=0"
    os.system(system_call)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--set_type', type=str, default='test')
    parser.add_argument('--internal_metric_name', type=str, default='R2Score_Exp')
    
    args = parser.parse_args()
    simulator_type, pow_type, cluster_type, hidden_size, hidden_layers, set_type = \
        args.simulator_type, args.pow_type, args.cluster_type, args.hidden_size, args.hidden_layers, args.set_type
    
    # Single Strategies
    print("Plotting for single strategies")
    Parallel(n_jobs=os.cpu_count())(
        delayed(task)(simulator_type, pow_type, cluster_type, set_type, hidden_size, hidden_layers, item,
                      args.internal_metric_name) for item in single_strategies_list
    )
    
    # Multiple Strategies
    print("Plotting for multiple strategies")
    system_call = \
        f"python -m plot_averages.multiple_strategies --simulator_type={simulator_type} --pow_type={pow_type} " + \
        f"--cluster_type={cluster_type} --dataset_type=not_null --mode=cl --batch_size=4096 --strategies {multiple_strategies} " + \
        f"--set_type={set_type} --show=0 --hidden_size={hidden_size} --hidden_layers={hidden_layers} --internal_metric_name={args.internal_metric_name}"
    if simulator_type == 'tglf':
        system_call = f"{system_call} --exclude_naive=1"
        if pow_type == 'lowpow':
            system_call = f"{system_call} --include_std=0"
    os.system(system_call)
