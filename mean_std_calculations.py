import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

from src.utils import *
from src.ex_post_tests import *


cl_strategies = [
    'Naive', 'FromScratchTraining', 'Cumulative',
    'Replay', 'PercentageReplay', 'EWC', 'EWCReplay',
    'MAS', 'MASReplay', 'GEM', 'GEMReplay', 'LFL', 'SI'
]

simulator_prefixes = {
    'qualikiz': [''],
    'tglf': ['TGLF/', 'TGLF/MSECOS 0.01 100/']
}

cl_extra_log_folders = [
    ['Base'], ['Base'], ['Base'], ['Buffer 500', 'Buffer 1000', 'Buffer 2000', 'Buffer 10000'],
    ['Percentage 5%', 'Percentage 10%'], ['Lambda 1', 'Lambda 10'], ['Lambda 10 Buffer 1000'],
    ['Lambda 10 Alpha 0,5', 'Lambda 1 Alpha 0,5'], ['Lambda 10 Alpha 0,5 Buffer 1000'],
    ['Patterns 100', 'Patterns 400', 'Patterns 1000', 'Patterns 2000'],
    ['Patterns 100 Buffer 1000'], ['Lambda 1', 'Lambda 10'], ['Lambda 0.1', 'Lambda 1']
]


al_cl_strategies = ['Naive', 'Replay', 'EWC', 'MAS', 'SI', 'GEM', 'Cumulative']

_al_cl_bmdal_strategies = ['random', 'maxdiag', 'maxdet', 'lcmd']
_al_cl_bmdal_initial_strategies = ['random', 'maxdiag']
_al_cl_bmdal_batch_size = 2000

_al_cl_bmdal_prefixes = []
for t in _al_cl_bmdal_initial_strategies:
    for s in _al_cl_bmdal_strategies:
        _al_cl_bmdal_prefixes.append(
            'AL(CL)/' + s + ' [rp, [512]] + ' + t + f'/Batch {_al_cl_bmdal_batch_size} full first set'
        )

al_cl_extra_log_folders = [
    _al_cl_bmdal_prefixes, # Naive
    [item + ' ' + 'Buffer 1000' for item in _al_cl_bmdal_prefixes], # Replay (1000)
    [item + ' ' + 'Lambda 10' for item in _al_cl_bmdal_prefixes], # EWC (10)
    [item + ' ' + 'Lambda 10 Alpha 0,5' for item in _al_cl_bmdal_prefixes], # MAAS (10, 0.5)
    [item + ' ' + 'Lambda 1' for item in _al_cl_bmdal_prefixes], # SI (1)
    [item + ' ' + 'Patterns 1000' for item in _al_cl_bmdal_prefixes], # GEM (200, 2500)
    _al_cl_bmdal_prefixes, # Cumulative
]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='cl')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--metric', type=str, default='Forgetting_Exp')
    parser.add_argument('--output_prefix', type=str, default='forgetting')
    parser.add_argument('--time_filename', type=str, default='timing.txt')
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--count', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'cl':
        strategies = cl_strategies
        extra_log_folders = cl_extra_log_folders
    elif args.mode == 'al_cl':
        strategies = al_cl_strategies
        extra_log_folders = al_cl_extra_log_folders
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    suffix = f'({args.batch_size} batch size) ({args.hidden_size} hidden size)'
    if args.hidden_layers != 2 or args.simulator_type == 'tglf':
        suffix = suffix + f' ({args.hidden_layers} hidden layers)'
    outputs = 'efe_efi_pfe_pfi'
    metric = args.metric
    output_filename = f'{args.output_prefix}_mean_std.csv'
    time_filename = args.time_filename
    times = {}

    current_simulator_prefixes = simulator_prefixes[args.simulator_type]
    tqdm_size = len(current_simulator_prefixes) * sum([len(folder) for folder in extra_log_folders])
    queue = tqdm(tqdm_size)
    
    for simulator_prefix in current_simulator_prefixes:
        for (strategy, folders) in zip(strategies, extra_log_folders):
            for folder in folders:
                extra_log_folder = simulator_prefix + folder + ' ' + suffix
                mean_std_df = mean_std_df_wrapper(
                    args.pow_type, args.cluster_type, args.dataset_type,
                    args.task, outputs, strategy, extra_log_folder,
                    metric=metric, count=args.count
                )
                if mean_std_df is not None:
                    log_folder = get_log_folder(
                        args.pow_type, args.cluster_type, args.task, args.dataset_type,
                        outputs, strategy, extra_log_folder, count=args.count, task_id=0,
                        simulator_type=args.simulator_type
                    )
                    mean_std_df['eval_exp'] = 10 * [0]
                    mean_std_df['training_exp'] = [i for i in range(10)]
                    mean_std_df.round(decimals=4).to_csv(os.path.join(log_folder, output_filename))
                    task_id = 0
                    num_tasks = 0
                    time_str = f'{strategy} - {simulator_prefix}/{folder}'
                    times[time_str] = 0
                    while True:
                        try:
                            log_folder = get_log_folder(
                                args.pow_type, args.cluster_type, args.task, args.dataset_type,
                                outputs, strategy, extra_log_folder, count=args.count,
                                task_id=task_id, simulator_type=args.simulator_type
                            )
                            num_tasks += 1
                            with open(os.path.join(log_folder, time_filename), 'r') as fp:
                                line = fp.readlines()[0][len('[run] Elapsed Time: '):]
                            hhmmss = [
                                int(round(float(val), 1)) for val in line.split(':')
                            ]
                            total_time = hhmmss[0] * 3600 + hhmmss[1] * 60 + hhmmss[2]
                            times[time_str] += total_time
                            task_id += 1
                        except ValueError as verr:
                            break
                    times[time_str] = times[time_str] / num_tasks
                queue.update(1)
    print("Times: ")
    print(json.dumps(times, indent=2))
