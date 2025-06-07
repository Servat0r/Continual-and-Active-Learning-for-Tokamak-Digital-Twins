# NOTE: QuaLiKiz-only!
import json
import numpy as np
from src.utils.logging import LoggingConfiguration
from src.ex_post_tests import get_mean_std_metric_values, get_training_times, get_num_epochs, \
    load_dataset_weights, extract_dataset_weights, computeR_from_config, computeT_from_config


simulator_type = "qualikiz"
pow_type = "mixed"
cluster_type = "beta_based" #"wmhd_based" # 
dataset_type = "not_null"
task = "regression"
hidden_size = 1024
hidden_layers = 2

strategies = [
    'Naive', 'Cumulative', 'FromScratchTraining', 'Replay', 'PercentageReplay', 'EWC', 'MAS', 'GEM', 'SI', 'LFL', 'EWCReplay', 'GEMReplay'
]

extra_log_folders = [
    ['Base'], ['Base'], ['Base'], ['Buffer 2000', 'Buffer 10000', 'Buffer 20000'], ['Percentage 5% Min 2000', 'Percentage 10% Min 10000'],
    ['Lambda 10'], ['Lambda 1 Alpha 0.0'], ['Patterns 1000'], ['Lambda 1'], ['Lambda 1'],
    ['Lambda 10 Buffer 1000', 'Lambda 1 Buffer 10000'],
    ['Patterns 1000 Buffer 10000']
]

plot_names = [
    [''], [''], [''], ['2000', '10000', '20000'], ['5%, 2000', '10%, 10000'],
    ['10'], ['1, 0'], ['1000'], ['1'], ['1'], ['10, 1000', '1, 10000'], ['1000, 10000']
]

print("[LOG] Imports done!")


try:
    absolute_weights = load_dataset_weights(
        simulator_type, pow_type, cluster_type, dataset_type, raw_or_final='final', task=task, weights_source='test'
    )
except FileNotFoundError:
    absolute_weights = extract_dataset_weights(
        simulator_type, pow_type, cluster_type, dataset_type, raw_or_final='final', task=task, weights_source='test'
    )

naive_config = LoggingConfiguration(
    pow_type, cluster_type, dataset_type, task, strategy='Naive',
    extra_log_folder='Base', simulator_type=simulator_type, hidden_size=hidden_size
)

cumulative_config = LoggingConfiguration(
    pow_type, cluster_type, dataset_type, task, strategy='Cumulative',
    extra_log_folder='Base', simulator_type=simulator_type, hidden_size=hidden_size
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

print("[LOG] Baselines computed!")

db_filename = f'db_qlk_{pow_type}_{cluster_type}.json' # NOTE: QuaLiKiz
#db_filename = f'db_tglf_{pow_type}_{cluster_type}.json' # NOTE: TGLF

with open(db_filename, 'w') as fp:
    records = [] # Database-like records
    for strategy, elf_set, plot_names_set in zip(strategies, extra_log_folders, plot_names):
        for extra_log_folder, plot_name in zip(elf_set, plot_names_set):
            try:
                strategy_config = LoggingConfiguration(
                    pow_type, cluster_type, dataset_type, task, strategy=strategy,
                    extra_log_folder=extra_log_folder, simulator_type=simulator_type,
                    hidden_size=hidden_size
                )

                #print(strategy, extra_log_folder, plot_name)

                strategy_values = get_mean_std_metric_values(
                    None, strategy_config.get_log_folder(), mean_filename='test_mean_values.csv', std_filename='test_std_values.csv',
                    metric='R2Score_Exp', absolute_weights=absolute_weights
                )

                strategy_R_values = computeR_from_config(strategy_config, naive_values, cumulative_values)
                strategy_times, strategy_total_time = get_training_times(strategy_config, num_tasks=4)
                strategy_epochs, _ = get_num_epochs(strategy_config, num_tasks=4)
                strategy_cumulative_times = np.cumsum(strategy_times)
                new_record = {
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
                #print(new_record)
                records.append(new_record)
            except FileNotFoundError:
                print(f"[LOG] Missing {strategy}-{extra_log_folder}-{plot_name}")
    json.dump(records, fp, indent=2)

print("[LOG] Database built!")
