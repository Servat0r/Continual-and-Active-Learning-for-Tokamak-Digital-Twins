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
        num_exp = config_parser.get_raw_config()['general']['num_campaigns']
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
            exp_log_dict = {'raw_metrics': {}, 'aggregated_metrics': {}}
            logging_config = get_logging_config_from_filepath(results[0]['log_folder']) # ?
            # Now add aggregated metrics to log dict
            strategy_times, strategy_total_time = get_training_times(logging_config, num_tasks=cmd_args.num_tasks)
            strategy_epochs, _ = get_num_epochs(logging_config, num_tasks=4)
            strategy_cumulative_times = np.cumsum(strategy_times)
            exp_log_dict['aggregated_metrics'].update({
                "times": strategy_times.round(4).tolist(),
                "total_time": strategy_total_time,
                "cumulative_times": strategy_cumulative_times.round(4).tolist(),
                "num_epochs": strategy_epochs.round(2).tolist()
            })
            db.add_or_update_logs(experiment_id, exp_log_dict)
            for set_type in ['eval', 'test']:
                file_paths = [
                    os.path.join(
                        result['log_folder'], f'{set_type}_results_experience.csv'
                    ) for result in results if result is not None
                ]
                if (len(file_paths) != len(results)):
                    raise RuntimeError(f"Something went wrong during training: {len(file_paths)} vs. {len(results)}")
                
                save_folder = results[0]['log_folder']
                dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths]
                # Efficient vectorized computation for mean and std across dataframes
                arr = np.stack([df.to_numpy(dtype=np.float32) for df in dfs])
                mean_arr = np.round(arr.mean(axis=0), decimals=4)
                std_arr = np.round(arr.std(axis=0), decimals=4)
                mean_df = pd.DataFrame(mean_arr, columns=dfs[0].columns)
                std_df = pd.DataFrame(std_arr, columns=dfs[0].columns)
                mean_df.to_csv(os.path.join(save_folder, f'{set_type}_mean_values.csv'), index=False)
                std_df.to_csv(os.path.join(save_folder, f'{set_type}_std_values.csv'), index=False)
                raw_columns = []
                cleaned_columns = []
                for col in mean_df.columns:
                    if col.endswith('_Exp'):
                        raw_columns.append(col)
                        cleaned_columns.append(f"{set_type.capitalize()}_{col[:-4]}")
                for raw_col, cleaned_col in zip(raw_columns, cleaned_columns):
                    exp_log_dict['raw_metrics'][cleaned_col] = {}
                    exp_log_dict['raw_metrics'][cleaned_col]['mean'] = mean_df[raw_col].to_list()
                    exp_log_dict['raw_metrics'][cleaned_col]['std'] = std_df[raw_col].to_list()
                # Now compute aggregated metrics
                try:
                    absolute_weights = load_dataset_weights(logging_config.scenario, raw_or_final='final', weights_source=set_type)
                except FileNotFoundError:
                    absolute_weights = extract_dataset_weights(logging_config.scenario, raw_or_final='final', weights_source=set_type)
                r2_strategy_values = get_mean_std_metric_values(
                    None, None, metric='R2Score_Exp', absolute_weights=absolute_weights,
                    mean_df=mean_df, std_df=std_df, num_exp=num_exp
                )
                rd_strategy_values = get_mean_std_metric_values(
                    None, None, metric='RelativeDistance_Exp', absolute_weights=absolute_weights,
                    mean_df=mean_df, std_df=std_df, num_exp=num_exp
                )
                exp_log_dict['aggregated_metrics'].update({
                    f"{set_type.capitalize()}_R2": {
                        'mean': r2_strategy_values['Mean R2Score_Exp'].to_list(),
                        'std': r2_strategy_values['Std R2Score_Exp'].to_list(),
                    },
                    f"{set_type.capitalize()}_RelativeDistance": {
                        'mean': rd_strategy_values['Mean RelativeDistance_Exp'].to_list(),
                        'std': rd_strategy_values['Std RelativeDistance_Exp'].to_list(),
                    },
                })
                db.add_or_update_logs(experiment_id, exp_log_dict)
                # Now it should check whether to insert derived metrics (R, time_ratios etc)
                size = len(r2_strategy_values['Mean R2Score_Exp'])
                if strategy['name'] == 'Naive':
                    exp_log_dict['aggregated_metrics'].update({
                        f"{set_type.capitalize()}_R": np.ones(size, dtype=np.float64).tolist(),
                        f"{set_type.capitalize()}_time_ratios": np.ones(size, dtype=np.float64).tolist()
                    })
                else:
                    if strategy['name'] == 'Cumulative':
                        exp_log_dict['aggregated_metrics'].update({
                            f"{set_type.capitalize()}_R": np.ones(size, dtype=np.float64).tolist()
                        })
                    else:
                        # R for other strategies
                        r_data = compute_derived_metrics(db, experiment_id, experiment_name, set_type, which='R')
                        exp_log_dict['aggregated_metrics'].update(r_data)
            if strategy['name'] != 'Naive':
                # time_ratios for Cumulative and other strategies
                t_data = compute_derived_metrics(db, experiment_id, experiment_name, None, which='time_ratios')
                exp_log_dict['aggregated_metrics'].update(t_data)
            # Finally, commit exp log dict to database
            exp_dict = db.add_or_update_logs(experiment_id, exp_log_dict)
            if exp_dict is None: # Failure in logs update
                stdout_debug_print(f"Failed to update logs for {experiment_name}", color='red')
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
    if _cleanup_aborted and (len(aborted_experiment_names) > 0):
        cleanup_aborted_experiments(db, targets=aborted_experiment_ids)
    elif _cleanup_tests and (len(test_names) > 0):
        cleanup_tests(db, targets=test_ids)
    elif _cleanup_all and (len(all_names) > 0):
        cleanup_all(db, targets=all_ids)
