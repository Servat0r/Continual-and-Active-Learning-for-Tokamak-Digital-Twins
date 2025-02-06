from typing import *

import json
import sys
import os

from joblib import Parallel, delayed

sys.path.append(os.path.dirname(__file__))  # Add src directory to sys.path

from src.utils import *
from src.configs import *
from src.run import *

if int(os.getenv('IGNORE_WARNINGS', '0')):
    import warnings
    warnings.filterwarnings("ignore")


def filtered_task_training_loop(
        config_data: str | dict[str, Any], task_id: int,
        redirect_stdout=True, extra_log_folder='',
        write_intermediate_models=False,
        plot_single_runs=False, tasks_list: list[int] = None,
):
    condition = (tasks_list is None) or (task_id in tasks_list)
    if condition:
        return task_training_loop(
            config_data, task_id, redirect_stdout, extra_log_folder,
            write_intermediate_models, plot_single_runs
        )
    else:
        debug_print(f"[red]Ignoring task {task_id} ...[/red]", file=STDOUT)
        return None


if __name__ == '__main__':
    # Config
    if len(sys.argv) < 2:
        print(f"[red]Usage[/red]: [cyan](python) {sys.argv[0]} configuration_file_name.json [OPTIONS] [/cyan]")
        print(f"[cyan]Options: [/cyan]")
        print(f"[cyan]--num_tasks=<int> (parallel execution on multiple model instances)[/cyan]")
        print(f"[cyan]--tasks=<list of int> (if you want to filter by tasks)[/cyan]")
        print(f"[cyan]-h[/cyan] or [cyan]--help[/cyan] (help messages)")
        sys.exit(1)

    cmd_arg_parser = build_argparser()
    # Parse arguments
    cmd_args = cmd_arg_parser.parse_args()
    test = cmd_args.test
    if test: sys.exit(0)
    config_file_path = cmd_args.config
    to_redirect_stdout = True if not cmd_args.no_redirect_stdout else False
    extra_log_folder = cmd_args.extra_log_folder or 'Base'
    write_intermediate_models = cmd_args.write_intermediate_models
    plot_single_runs = cmd_args.plot_single_runs
    if cmd_args.num_tasks <= 0:
        num_jobs = os.cpu_count() // 2
    else:
        num_jobs = cmd_args.num_tasks
    # Config data preprocessing
    config_data = json.load(open(config_file_path))
    if not isinstance(config_data['strategy'], list):
        config_data['strategy'] = [config_data['strategy']]
    tasks_list = cmd_args.tasks
    if tasks_list is not None:
        debug_print(f"[red]Tasks {tasks_list} will be run ...[/red]", file=STDOUT)
    for strategy in config_data['strategy']:
        ignore_strategy = strategy.get('ignore', False)
        if ignore_strategy:
            debug_print(f"[red]Ignoring strategy: {strategy['name']} ... [/red]", file=STDOUT)
            continue
        else:
            debug_print(f"[red]Running strategy: {strategy['name']} ... [/red]", file=STDOUT)
        single_config_data = config_data.copy()
        single_config_data['strategy'] = strategy
        if num_jobs > 1:
            task_ids = range(num_jobs)
            results = \
                Parallel(n_jobs=num_jobs)(
                    delayed(task_training_loop)(
                        single_config_data, task_id, to_redirect_stdout, extra_log_folder,
                        write_intermediate_models, plot_single_runs
                    ) for task_id in task_ids
                )
        else:
            results = [
                task_training_loop(
                    single_config_data, 0, redirect_stdout=to_redirect_stdout,
                    extra_log_folder=extra_log_folder,
                    write_intermediate_models=write_intermediate_models,
                    plot_single_runs=plot_single_runs
                )
            ]
        # Plot means and standard deviations
        if num_jobs > 1:
            file_paths = [
                os.path.join(result['log_folder'], 'eval_results_experience.csv') for result in results if result is not None
            ]
            if (len(file_paths) != len(results)) and (len(tasks_list) < num_jobs):
                raise RuntimeError(f"Something went wrong during training: {len(file_paths)} vs. {len(results)}")
            save_folder = os.path.dirname(file_paths[0])
            # Save csv files for mean and std values
            get_means_std_over_evaluation_experiences_multiple_runs(
                file_paths, os.path.join(save_folder, 'mean_values.csv'), os.path.join(save_folder, 'std_values.csv')
            )
            # Plot mean and std values
            task = results[0]['task']
            metric_list = get_metric_names_list(task)
            title_list = get_title_names_list(task)
            ylabel_list = get_ylabel_names_list(task)
            if results[0]['is_joint_training']:
                mean_std_evaluation_experiences_plots(
                    file_paths, metric_list, title_list, ylabel_list, start_exp=0, end_exp=0, num_exp=1,
                )
            else:
                mean_std_evaluation_experiences_plots(file_paths, metric_list, title_list, ylabel_list)
