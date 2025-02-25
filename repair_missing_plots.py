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

args = parser.parse_args()

logging_config = LoggingConfiguration(
    pow_type=args.pow_type, cluster_type=args.cluster_type, dataset_type=args.dataset_type,
    task=args.task, strategy=args.strategy, extra_log_folder=args.extra_log_folder,
    simulator_type=args.simulator_type, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
    batch_size=args.batch_size, active_learning=False
)

file_paths = []
task_id = 0

while True:
    try:
        log_folder = logging_config.get_log_folder(count=args.count, task_id=task_id, suffix=True)
        file_paths.append(os.path.join(log_folder, 'eval_results_experience.csv'))
        task_id += 1
    except:
        break

stdout_debug_print(f'Detected {task_id} tasks', color='red')

if len(file_paths) == 0:
    raise RuntimeError("No evaluation result files found!")
save_folder = os.path.dirname(file_paths[0])

get_means_std_over_evaluation_experiences_multiple_runs(
    file_paths,
    os.path.join(save_folder, 'mean_values.csv'),
    os.path.join(save_folder, 'std_values.csv')
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
        start_exp=0, end_exp=0, num_exp=1
    )
else:
    mean_std_evaluation_experiences_plots(
        file_paths, metric_list, title_list, ylabel_list
    )
