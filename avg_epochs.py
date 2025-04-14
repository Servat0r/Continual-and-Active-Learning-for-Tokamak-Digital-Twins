import numpy as np
import pandas as pd

from src.utils import *
from src.ex_post_tests import *
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--strategy', type=str, default='Naive')
    parser.add_argument('--extra_log_folder', type=str, default='Base')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--mode', type=str, default='cl')
    parser.add_argument('--al_method', type=str, default='random_sketch_grad')
    parser.add_argument('--al_batch_size', type=int, default=256)
    parser.add_argument('--al_max_batch_size', type=int, default=1024)
    parser.add_argument('--full_first_set', type=bool, default=False)
    parser.add_argument('--reload_weights', type=bool, default=False)
    parser.add_argument('--downsampling', type=float, default=0.5)
    parser.add_argument('--ncampaigns', type=int, default=10)

    args = parser.parse_args()

    logging_config = LoggingConfiguration(
        pow_type=args.pow_type, cluster_type=args.cluster_type, dataset_type=args.dataset_type,
        task=args.task, strategy=args.strategy, extra_log_folder=args.extra_log_folder,
        simulator_type=args.simulator_type, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
        batch_size=args.batch_size, active_learning=(args.mode == 'al_cl'), al_method=args.al_method,
        al_batch_size=args.al_batch_size, al_max_batch_size=args.al_max_batch_size,
        al_full_first_set=args.full_first_set, al_reload_weights=args.reload_weights,
        al_downsampling_factor=args.downsampling
    )
    folder_paths = []
    ntasks = 0
    if args.mode == 'cl':
        while True:
            try:
                path = logging_config.get_log_folder(count=-1, task_id=ntasks)
                #path = f'{path}/training_results_epoch.csv'
                folder_paths.append(path)
                ntasks += 1
            except:
                break
    else:
        while True:
            try:
                path = logging_config.get_log_folder(count=-1, task_id=ntasks)
                #path = f'{path}/training_results_epoch.csv'
                folder_paths.append(path)
                ntasks += 1
            except:
                break
    dfs = []
    exps = list(range(args.ncampaigns))
    for path in folder_paths:
        # Time Data
        times = []
        df = pd.read_csv(f"{path}/training_results_epoch.csv")
        for campaign in range(args.ncampaigns):
            campaign_df = df[df.training_exp == campaign]
            num_epochs = len(campaign_df)
            times.append(num_epochs)
        time_df = pd.DataFrame({'Experience': exps, 'Epochs': times})
        dfs.append(time_df)
    
    # #Epochs Calculation
    mean_df = sum(dfs) / len(dfs)
    arrs = [df['Epochs'].to_numpy() for df in dfs]
    std_arr = np.std(arrs, axis=0)
    #print(arrs, std_arr)
    std_df = pd.DataFrame({'Experience': exps, 'Epochs': std_arr})
    final_df = pd.DataFrame({'Experience': exps, 'Mean Epochs': mean_df['Epochs'], 'Std Epochs': std_df['Epochs']})
    avg_num_epochs = final_df['Mean Epochs'].mean()
    print(final_df)
    print(f"Average #epochs = {avg_num_epochs}")
