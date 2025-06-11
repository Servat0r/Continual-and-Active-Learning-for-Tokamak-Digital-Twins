# NOTE: QuaLiKiz-only!
import json, os
import numpy as np
from itertools import product
from src.utils.datasets import QUALIKIZ_HIGHPOW_OUTPUTS
from src.utils.scenarios import *
from src.utils.logging import LoggingConfiguration
from src.ex_post_tests import *
from argparse import ArgumentParser


cl_hidden_size = 1024
al_cl_hidden_size = 256
hidden_layers = 2


def get_common_params(mode: str):
    if mode.upper() == "CL":
        strategies = [
            'Naive', 'Cumulative', 'FromScratchTraining', 'Replay', 'PercentageReplay',
            'EWC', 'MAS', 'GEM', 'SI', 'LFL', 'EWCReplay', 'GEMReplay'
        ]

        extra_log_folders = [
            ['Base'], ['Base'], ['Base'], ['Buffer 2000', 'Buffer 10000', 'Buffer 20000'],
            ['Percentage 5% Min 2000', 'Percentage 10% Min 10000'],
            ['Lambda 10'], ['Lambda 1 Alpha 0.0'], ['Patterns 400', 'Patterns 1000', 'Patterns 1024'],
            ['Lambda 1'], ['Lambda 1'], ['Lambda 10 Buffer 1000', 'Lambda 1 Buffer 10000'],
            ['Patterns 1000 Buffer 10000']
        ]

        plot_names = [
            [''], [''], [''], ['2000', '10000', '20000'], ['5%, 2000', '10%, 10000'],
            ['10'], ['1, 0'], ['400', '1000', '1024'], ['1'], ['1'], ['10, 1000', '1, 10000'], ['1000, 10000']
        ]
    else:
        strategies = [
            'Naive', 'Cumulative', 'Replay', 'EWCReplay', 'GEM'
        ]

        extra_log_folders = [
            ['Base'], ['Base'], ['Buffer 2000'],
            ['Lambda 1 Buffer 2000'], ['Patterns 400', 'Patterns 1024']
        ]

        plot_names = [
            [''], [''], ['2000'], ['1, 2000'], ['400', '1024']
        ]
    return strategies, extra_log_folders, plot_names


### Active Learning Config Parameters
batch_sizes = [256]
max_batch_sizes = [1024]
reload_initial_weights = [False]
standard_methods = ['random_sketch_grad', 'batchbald', 'badge', 'lcmd_sketch_grad']
full_first_sets = [True]
first_set_sizes = [5120]
downsampling_factors = [0.5]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--overwrite', type=str) # "true" or "false"
    parser.add_argument('--mode', type=str, default='CL') # 'CL' or 'AL(CL)'
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='mixed')
    parser.add_argument('--cluster_type', type=str, default='beta_based')
    parser.add_argument('--metric', type=str, default='R2',
                      help='Metric to plot (R2, R, times, cumulative_times, time_ratios, or num_epochs)')
    args = parser.parse_args()

    config = ScenarioConfig(
        simulator_type=args.simulator_type,
        pow_type=args.pow_type,
        cluster_type=args.cluster_type,
        dataset_type='not_null',
        task='regression',
        outputs=QUALIKIZ_HIGHPOW_OUTPUTS
    )
    db_filename = get_db_filename(config)
    records = []
    if os.path.exists(db_filename) and (args.overwrite.lower() == "false"):
        with open(db_filename, 'r') as fp:
            records = json.load(fp)

    try:
        absolute_weights = load_dataset_weights(config, raw_or_final='final', weights_source='test')
    except FileNotFoundError:
        absolute_weights = extract_dataset_weights(config, raw_or_final='final', weights_source='test')
    
    strategies, extra_log_folders, plot_names = get_common_params(args.mode)
    if args.mode.upper() == 'CL':
        naive_config = LoggingConfiguration(
            config, strategy='Naive', extra_log_folder='Base', hidden_size=cl_hidden_size, hidden_layers=hidden_layers
        )

        cumulative_config = LoggingConfiguration(
            config, strategy='Cumulative', extra_log_folder='Base', hidden_size=cl_hidden_size, hidden_layers=hidden_layers
        )

        naive_values = get_mean_std_metric_values(
            None, naive_config.get_log_folder(), mean_filename='test_mean_values.csv', std_filename='test_std_values.csv',
            metric='R2Score_Exp', absolute_weights=absolute_weights
        )

        cumulative_values = get_mean_std_metric_values(
            None, cumulative_config.get_log_folder(), mean_filename='test_mean_values.csv', std_filename='test_std_values.csv',
            metric='R2Score_Exp', absolute_weights=absolute_weights
        )

        naive_times, naive_total_time = get_training_times(naive_config, num_tasks=4)
        naive_cumulative_times = np.cumsum(naive_times)
        cumulative_times, cumulative_total_time = get_training_times(cumulative_config, num_tasks=4)

        with open(db_filename, 'w') as fp:
            for strategy, elf_set, plot_names_set in zip(strategies, extra_log_folders, plot_names):
                for extra_log_folder, plot_name in zip(elf_set, plot_names_set):
                    try:
                        strategy_config = LoggingConfiguration(
                            config, strategy=strategy, extra_log_folder=extra_log_folder,
                            hidden_size=cl_hidden_size, hidden_layers=hidden_layers
                        )

                        strategy_values = get_mean_std_metric_values(
                            None, strategy_config.get_log_folder(), mean_filename='test_mean_values.csv',
                            std_filename='test_std_values.csv', metric='R2Score_Exp', absolute_weights=absolute_weights
                        )

                        strategy_R_values = computeR_from_config(strategy_config, naive_values, cumulative_values)
                        strategy_times, strategy_total_time = get_training_times(strategy_config, num_tasks=4)
                        strategy_epochs, _ = get_num_epochs(strategy_config, num_tasks=4)
                        strategy_cumulative_times = np.cumsum(strategy_times)
                        new_record = {
                            "mode": "CL",
                            **config.to_dict(),
                            "hidden_size": cl_hidden_size,
                            "hidden_layers": hidden_layers,
                            "strategy": strategy,
                            "extra_log_folder": extra_log_folder,
                            "plot_name": plot_name,
                            "R2": strategy_values['Mean R2Score_Exp'].to_numpy().round(3).tolist(),
                            "R": strategy_R_values.round(3).tolist(),
                            "times": strategy_times.round(3).tolist(),
                            "cumulative_times": strategy_cumulative_times.round(3).tolist(),
                            "total_time": round(strategy_total_time, 3),
                            "time_ratios": (strategy_cumulative_times / naive_cumulative_times).round(3).tolist(),
                            "final_time_ratio": round(strategy_total_time / naive_total_time, 3),
                            "num_epochs": strategy_epochs.tolist()
                        }
                        records.append(new_record)
                    except FileNotFoundError:
                        print(f"[LOG] Missing {strategy}-{extra_log_folder}-{plot_name}")
            json.dump(records, fp, indent=2)
    elif args.mode.upper() in ['AL(CL)', 'AL_CL', 'ACL', 'CLAEA']:
        print(f"{args.mode}")
        param_combinations = list(product(
            batch_sizes,
            max_batch_sizes,
            reload_initial_weights,
            standard_methods,
            full_first_sets,
            first_set_sizes,
            downsampling_factors
        ))
        with open(db_filename, 'w') as fp:
            for strategy, elf_set, plot_names_set in zip(strategies, extra_log_folders, plot_names):
                for extra_log_folder, plot_name in zip(elf_set, plot_names_set):
                    for batch_size, max_batch_size, reload_weights, method, full_first, \
                        first_size, down_factor in param_combinations:
                        try:
                            al_config = ActiveLearningConfig(
                                framework='bmdal',
                                batch_size=batch_size,
                                max_batch_size=max_batch_size,
                                reload_initial_weights=reload_weights,
                                standard_method=method,
                                full_first_set=full_first,
                                first_set_size=first_size,
                                downsampling_factor=down_factor
                            )
                            strategy_config = LoggingConfiguration(
                                config, strategy=strategy, extra_log_folder=extra_log_folder,
                                hidden_size=al_cl_hidden_size, hidden_layers=hidden_layers,
                                active_learning=True, al_config=al_config
                            )
                            log_folder = strategy_config.get_log_folder()
                            strategy_values = get_mean_std_metric_values(
                                None, log_folder, mean_filename='test_mean_values.csv',
                                std_filename='test_std_values.csv', metric='R2Score_Exp',
                                absolute_weights=absolute_weights
                            )
                            strategy_times, strategy_total_time = get_training_times(strategy_config, num_tasks=4)
                            strategy_epochs, _ = get_num_epochs(strategy_config, num_tasks=4)
                            strategy_cumulative_times = np.cumsum(strategy_times)
                            new_record = {
                                "mode": "AL(CL)",
                                **config.to_dict(),
                                "hidden_size": al_cl_hidden_size,
                                "hidden_layers": hidden_layers,
                                "strategy": strategy,
                                "extra_log_folder": extra_log_folder,
                                **al_config.to_dict(),
                                "plot_name": plot_name,
                                "R2": strategy_values['Mean R2Score_Exp'].to_numpy().round(3).tolist(),
                                "times": strategy_times.round(3).tolist(),
                                "cumulative_times": strategy_cumulative_times.round(3).tolist(),
                                "total_time": round(strategy_total_time, 3),
                                "num_epochs": strategy_epochs.tolist()
                            }
                            records.append(new_record)
                        except FileNotFoundError:
                            print(f"[LOG] Missing {strategy}-{plot_name}-{al_config.standard_method}")
            json.dump(records, fp, indent=2)
