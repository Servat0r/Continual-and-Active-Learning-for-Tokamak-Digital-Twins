from typing import *

import traceback
import json, sys, os
import numpy as np
import pandas as pd

from time import sleep
from joblib import Parallel, delayed
from types import MappingProxyType

sys.path.append(os.path.dirname(__file__))

from src.utils import *
from src.configs import *
from src.database import *
from src.run import *

if int(os.getenv('IGNORE_WARNINGS', '0')):
    import warnings
    warnings.filterwarnings("ignore")


ConfigParser.__standardizer_dict__ = MappingProxyType(ConfigParser.__standardizer_dict__)
ConfigParser.__parsing_dict__ = MappingProxyType(ConfigParser.__parsing_dict__)

db = get_db()


if __name__ == '__main__':
    cmd_arg_parser = build_argparser()
    # Parse arguments
    cmd_args = cmd_arg_parser.parse_args()
    config_file_path = cmd_args.config
    to_redirect_stdout = True if not cmd_args.no_redirect_stdout else False
    extra_log_folder = cmd_args.extra_log_folder or 'Base'
    write_intermediate_models = cmd_args.write_intermediate_models
    plot_single_runs = cmd_args.plot_single_runs
    _cleanup_aborted = cmd_args.cleanup_aborted
    _cleanup_tests = cmd_args.cleanup_tests
    _cleanup_all = cmd_args.cleanup_all
    if (cmd_args.num_tasks <= 0) or (cmd_args.num_tasks is None):
        num_jobs = os.cpu_count() // 2
    else:
        num_jobs = cmd_args.num_tasks
    # Config data preprocessing
    config_data = json.load(open(config_file_path))
    if not isinstance(config_data['strategy'], list):
        config_data['strategy'] = [config_data['strategy']]
    aborted_experiment_ids = [] # List of all aborted experiments ids
    aborted_experiment_names = [] # List of all aborted experiments names
    test_ids = [] # List of all test ids
    test_names = [] # List of all test names
    all_ids = [] # List of all ids
    all_names = [] # List of all names
    is_test = bool(cmd_args.is_test or False)
    for strategy in config_data['strategy']:
        ignore_strategy = strategy.get('ignore', False)
        if ignore_strategy:
            debug_print(f"[red]Ignoring strategy: {strategy['name']} ... [/red]", file=STDOUT)
            continue
        else:
            debug_print(f"[red]Running strategy: {strategy['name']} ... [/red]", file=STDOUT)
        single_config_data = config_data.copy()
        single_config_data['strategy'] = strategy

        config_parser = ConfigParser(single_config_data, task_id=0) # <== TODO Fix this task_id requirement!
        config_parser.load_config()
        print(f"[main] Configuration standardized: {config_parser.is_standardized()}")
        #print(json.dumps(config_parser.raw_config, indent=2))
        num_tasks = cmd_args.num_tasks
        experiment_id, experiment_name = config2db(config_parser.raw_config, db, num_tasks, is_test)
        all_ids.append(experiment_id)
        all_names.append(experiment_name)
        if is_test:
            test_ids.append(experiment_id)
            test_names.append(experiment_name)
        # Now experiment has "init" status
        try:
            if num_jobs > 1:
                task_ids = range(num_jobs)
                results = \
                    Parallel(n_jobs=num_jobs)(
                        delayed(task_training_loop)(
                            single_config_data, task_id, experiment_id, experiment_name,
                            to_redirect_stdout, extra_log_folder, write_intermediate_models,
                            plot_single_runs, db.db_file
                        ) for task_id in task_ids
                    )
            else:
                results = [
                    task_training_loop(
                        single_config_data, 0, experiment_id, experiment_name,
                        to_redirect_stdout, extra_log_folder, write_intermediate_models,
                        plot_single_runs, db.db_file
                    )
                ]
            # Plot means and standard deviations
            if num_jobs > 1:
                for set_type in ['eval', 'test']:
                    file_paths = [
                        os.path.join(
                            result['log_folder'], f'{set_type}_results_experience.csv'
                        ) for result in results if result is not None
                    ]
                    if (len(file_paths) != len(results)):
                        raise RuntimeError(f"Something went wrong during training: {len(file_paths)} vs. {len(results)}")
                    save_folder = os.path.dirname(file_paths[0])
                    dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths]
                    columns = dfs[0].columns
                    mean_df = pd.DataFrame(columns=columns)
                    std_df = pd.DataFrame(columns=columns)
                    for column in columns:
                        values = [df[column].to_numpy(dtype=np.float32) for df in dfs]
                        arr = np.round(np.vstack(values), decimals=8)
                        mean_df[column] = arr.mean(axis=0)
                        std_df[column] = arr.std(axis=0)
                    mean_df.to_csv(os.path.join(save_folder, f'{set_type}_mean_values.csv'), index=False)
                    std_df.to_csv(os.path.join(save_folder, f'{set_type}_std_values.csv'), index=False)
                    # Plot mean and std values
                    task = results[0]['task']
                    metric_list = get_metric_names_list(task)
                    title_list = get_title_names_list(task)
                    ylabel_list = get_ylabel_names_list(task)
                    if results[0]['is_joint_training']:
                        mean_std_evaluation_experiences_plots(
                            file_paths, metric_list, title_list, ylabel_list,
                            start_exp=0, end_exp=0, num_exp=1, set_type=set_type
                        )
                    else:
                        mean_std_evaluation_experiences_plots(
                            file_paths, metric_list, title_list, ylabel_list, set_type=set_type
                        )
        except (KeyboardInterrupt, Exception) as ex:
            stdout_debug_print(f"Caught exception: {ex}", color='red')
            traceback.print_exc() # Print traceback
            aborted = 0
            while aborted == 0:
                aborted, _ = db.set_any_to_aborted([experiment_id])
                if aborted > 0:
                    stdout_debug_print(f"Experiment {experiment_name} was aborted", color='green')
                    aborted_experiment_ids.append(experiment_id)
                    aborted_experiment_names.append(experiment_name)
                else:
                    sleep(0.002)
            break # break from strategy for-loop
        finally:
            exp_dict = db.read_record(Experiment, experiment_id, as_dict=True)
            while exp_dict['status'] == 'running':
                finished, _ = db.set_running_to_finished([experiment_id])
                if finished > 0:
                    stdout_debug_print(f"Experiment {experiment_name} was successfully completed", color='green')
                else:
                    sleep(0.002)
                exp_dict = db.read_record(Experiment, experiment_id, as_dict=True)
    print(f"Aborted Experiments: {aborted_experiment_names}")
    print(f"Test Experiments: {test_names}")
    print(f"All Experiments: {all_names}")
    if _cleanup_aborted:
        cleanup_aborted_experiments(db, aborted_experiment_ids)
    elif _cleanup_tests:
        cleanup_tests(db, test_ids)
    elif _cleanup_all:
        cleanup_all(db, all_ids)
