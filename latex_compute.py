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
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--set_type', type=str, default='test')
    parser.add_argument('--time_type', type=str, default='training')

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
    last_exp = exps[args.index]
    set_type = args.set_type
    # R2Score
    mean_std_df = mean_std_df_wrapper(
        logging_config, f'{set_type}_mean_values.csv', f'{set_type}_std_values.csv',
        'R2Score_Exp', count=-1, include_future_experiences=False
    )
    last_row = mean_std_df[mean_std_df.Experience == last_exp]
    mean_r2_score = last_row['Mean R2Score_Exp'].item()
    std_r2_score = last_row['Std R2Score_Exp'].item()
    # Forgetting
    mean_std_df = mean_std_df_wrapper(
        logging_config, f'{set_type}_mean_values.csv', f'{set_type}_std_values.csv',
        'Forgetting_Exp', count=-1, include_future_experiences=False
    )
    last_row = mean_std_df[mean_std_df.Experience == last_exp]
    mean_forgetting = last_row['Mean Forgetting_Exp'].item()
    std_forgetting = last_row['Std Forgetting_Exp'].item()

    maxf_index, maxf_value = get_max_metric_value(folder_paths[0], f'{set_type}_mean_values.csv', f'{set_type}_std_values.csv')
    
    if args.time_type == 'training':
        for path in folder_paths:
            # Time Data
            times = []
            df = pd.read_csv(f"{path}/training_results_epoch.csv")
            for campaign in range(args.ncampaigns):
                campaign_df = df[df.training_exp == campaign]
                total_time = campaign_df['Time_Epoch'].sum()
                times.append(total_time)
            time_df = pd.DataFrame({'Experience': exps, 'Time': times})
            dfs.append(time_df)
        
        # Time Calculation
        mean_df = sum(dfs) / len(dfs)
        arrs = [df['Time'].to_numpy() for df in dfs]
        std_arr = np.std(arrs, axis=0)
        #print(arrs, std_arr)
        std_df = pd.DataFrame({'Experience': exps, 'Time': std_arr})
        final_df = pd.DataFrame({'Experience': exps, 'Mean Time': mean_df['Time'], 'Std Time': std_df['Time']})

        # Table Entry
        if args.mode == 'cl':
            print(
                f"$\\mathbf{{{mean_r2_score:.3f}\\pm{std_r2_score:.3f}}}$ & ${mean_forgetting:.3f}\\pm{std_forgetting:.3f}$ & " + \
                f"${maxf_value:.3f}$ (${maxf_index}$) & $\\mathbf{{{final_df['Mean Time'].sum():.3f}}}$"
            )
        else:
            print(
                f"${mean_r2_score:.3f}\\pm{std_r2_score:.3f}$ & ${final_df['Mean Time'].sum():.3f}$"
            )
    elif args.time_type == 'total':
        for path in folder_paths:
            with open(f"{path}/timing.txt", 'r') as fp:
                lines = fp.readlines()
            line_data = lines[0].strip()[len("[run] Elapsed Time: "):].split('.')
            hms_data = line_data[0].split(':')
            hh, mm, ss = int(hms_data[0]),int(hms_data[1]), int(hms_data[2])
            ff = round(float("0." + line_data[1]), 3)
            time_datum = 3600 * hh + 60 * mm + ss + ff
            dfs.append(time_datum)
        mean_time, std_time = round(sum(dfs) / len(dfs), 3), round(np.std(np.array(dfs)).item(), 3)
        # Table Entry
        if args.mode == 'cl':
            print(
                f"$\\mathbf{{{mean_r2_score:.3f}\\pm{std_r2_score:.3f}}}$ & ${mean_forgetting:.3f}\\pm{std_forgetting:.3f}$ & " + \
                f"${maxf_value:.3f}$ (${maxf_index}$) & $\\mathbf{{{mean_time}}}$"
            )
        else:
            print(
                f"${mean_r2_score:.3f}\\pm{std_r2_score:.3f}$ & ${mean_time}$"
            )
    else:
        raise ValueError(f"Invalid time_type = \"{args.time_type}\"")
