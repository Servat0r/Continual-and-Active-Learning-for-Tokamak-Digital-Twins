import os
from joblib import Parallel, delayed

from src.utils import *
from src.ex_post_tests import *
from argparse import ArgumentParser


al_methods = ['batchbald', 'badge']
strategies = ['Naive', 'Replay', 'EWCReplay', 'GEM', 'GEM', 'Cumulative']
#extra_log_folders=['Base', '"Buffer 2000"', '"Lambda 1 Buffer 2000"', '"Patterns 400"', '"Patterns 1024"', 'Base']
def get_extra_log_folders(simulator_type, pow_type):
    if simulator_type == 'qualikiz':
        if pow_type == 'lowpow':
            extra_log_folders=[
                'Base', '"Buffer 2500"', '"Lambda 1 Buffer 2000"', '"Patterns 200"', '"Patterns 1024"', 'Base'
            ]
        else:
            extra_log_folders=[
                'Base', '"Buffer 2000"', '"Lambda 1 Buffer 2000"', '"Patterns 400"', '"Patterns 1024"', 'Base'
            ]
    else:
        extra_log_folders=[
            'Base', '"Buffer 3000"', '"Lambda 1 Buffer 3000"', '"Patterns 300"', '"Patterns 1024"', 'Base'
        ]
    return extra_log_folders


BATCH_SIZES = [(256, 1024), (512, 1024)]
HIDDEN_SIZE = 256

def generator(strategies, extra_log_folders):
    for batch_sizes in BATCH_SIZES:
        for strategy, extra_log_folder in zip(strategies, extra_log_folders):
            yield strategy, batch_sizes, extra_log_folder


def task(simulator_type, pow_type, cluster_type, set_type, hidden_size, item, batch_sizes, extra_log_folder):
    print(f"Plotting for {item} and {batch_sizes} ...")
    al_batch_size, al_max_batch_size = batch_sizes
    system_call = \
        f"python -m plot_averages.multiple_methods --mode=al_cl --simulator_type={simulator_type} --pow_type={pow_type} " + \
        f"--cluster_type={cluster_type} --dataset_type=not_null --batch_size=4096 --strategy {item} --al_batch_size={al_batch_size} " + \
        f"--al_max_batch_size={al_max_batch_size} --set_type={set_type} --show=0 --hidden_size={hidden_size} --hidden_layers=2 " + \
        f"--al_methods batchbald badge --full_first_set=1 --extra_log_folder={extra_log_folder}"
    os.system(system_call)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--set_type', type=str, default='test')
    
    args = parser.parse_args()
    simulator_type, pow_type, cluster_type, hidden_size, hidden_layers, set_type = \
        args.simulator_type, args.pow_type, args.cluster_type, args.hidden_size, args.hidden_layers, args.set_type
    
    # Single Strategies
    print("Plotting for single strategies")
    Parallel(n_jobs=os.cpu_count())(
        delayed(task)(simulator_type, pow_type, cluster_type, set_type, hidden_size,
        strategy, batch_sizes, extra_log_folder) for strategy, batch_sizes, extra_log_folder \
        in generator(strategies, get_extra_log_folders(simulator_type, pow_type))
    )
