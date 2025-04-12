from typing import *

import json
import sys
import os

from joblib import Parallel, delayed

sys.path.append(os.path.dirname(__file__))  # Add src directory to sys.path

from src.utils import *
from src.configs import *
from src.run import *


import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate mean/std plots for missing cases')
parser.add_argument('--pow_type', type=str, default='highpow', help='Power type')
parser.add_argument('--cluster_type', type=str, default='tau_based', help='Cluster type')
parser.add_argument('--dataset_type', type=str, default='not_null', help='Dataset type')
parser.add_argument('--simulator_type', type=str, default='qualikiz', help='Simulator type')
parser.add_argument('--extra_log_folder', type=str, default='Base', help='Extra log folder path')

parser.add_argument('--task', type=str, default='regression', help='Task type')
parser.add_argument('--strategy', type=str, default='Naive', help='Strategy name')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size')
parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--count', type=int, default=-1, help='Run ordinal number')
parser.add_argument('--set_type', type=str, default='eval', help='Set type: either "eval" or "test"')
parser.add_argument('--mode', type=str, default='cl', help='Mode: either "cl" or "al_cl"')
parser.add_argument(
    '--al_method', type=str, default='random_sketch_grad', help='Single AL method to use (only for al_cl mode)'
)
parser.add_argument('--al_batch_size', type=int, default=128, help='Batch size for active learning')
parser.add_argument('--al_max_batch_size', type=int, default=2048, help='Maximum batch size for active learning')
parser.add_argument('--full_first_set', type=bool, default=False, help='Whether to use full first set in active learning')
parser.add_argument(
    '--reload_weights', type=bool, default=False, help='Whether to reload weights between active learning iterations'
)
parser.add_argument('--downsampling', type=float, default=0.5, help='Downsampling ratio for active learning')


args = parser.parse_args()
is_active_learning = args.mode == 'al_cl'
logging_config = LoggingConfiguration(
    pow_type=args.pow_type, cluster_type=args.cluster_type, dataset_type=args.dataset_type,
    task=args.task, strategy=args.strategy, extra_log_folder=args.extra_log_folder,
    simulator_type=args.simulator_type, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
    batch_size=args.batch_size, active_learning=is_active_learning, al_batch_size=args.al_batch_size,
    al_max_batch_size=args.al_max_batch_size, al_method=args.al_method, al_full_first_set=args.full_first_set,
    al_reload_weights=args.reload_weights, al_downsampling_factor=args.downsampling
)

file_paths = []
task_id = 0
set_type = args.set_type

while True:
    try:
        log_folder = logging_config.get_log_folder(count=args.count, task_id=task_id, suffix=True)
        file_paths.append(os.path.join(log_folder, f'{set_type}_results_experience.csv'))
        task_id += 1
    except:
        break

stdout_debug_print(f'Detected {task_id} tasks', color='red')

if len(file_paths) == 0:
    raise RuntimeError("No evaluation result files found!")
save_folder = os.path.dirname(file_paths[0])

get_means_std_over_evaluation_experiences_multiple_runs(
    file_paths,
    os.path.join(save_folder, f'{set_type}_mean_values.csv'),
    os.path.join(save_folder, f'{set_type}_std_values.csv')
)

# Get metric names and plot
metric_list = get_metric_names_list(args.task)
title_list = get_title_names_list(args.task)
ylabel_list = get_ylabel_names_list(args.task)

# Check if joint training by looking at number of experiences
import pandas as pd
first_file_data = pd.read_csv(file_paths[0])
is_joint_training = len(first_file_data['training_exp'].unique()) == 1

if is_joint_training:
    mean_std_evaluation_experiences_plots(
        file_paths, metric_list, title_list, ylabel_list,
        start_exp=0, end_exp=0, num_exp=1, set_type=set_type
    )
else:
    mean_std_evaluation_experiences_plots(
        file_paths, metric_list, title_list, ylabel_list, set_type=set_type
    )
