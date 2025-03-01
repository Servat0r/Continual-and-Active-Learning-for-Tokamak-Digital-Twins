import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.logging import LoggingConfiguration
from src.ex_post_tests import *
from plot_averages.common import common_parser_build


parser = common_parser_build(with_strategies=False)
parser.add_argument(
    "--strategy", type=str, default='Naive', help='CL or AL(CL) strategy.'
)
parser.add_argument("--task_id", type=int, default=0, help="Task ID")
parser.add_argument('--extra_log_folder', type=str, default='Base')
parser.add_argument(
    "--window_size", type=int, default=1,
    help="Size of the window over which to compute the (smoothed) average per epoch"
)
parser.add_argument("--start_exp", type=int, default=0, help="Starting Experience")
parser.add_argument("--end_exp", type=int, default=9, help="Ending Experience")

if __name__ == '__main__':
    args = parser.parse_args()
    pow_type, cluster_type, dataset_type, task, simulator_type = \
        args.pow_type, args.cluster_type, args.dataset_type, args.task, args.simulator_type
    
    window_size = args.window_size
    if not (window_size % 2): # window_size is even
        raise RuntimeError(f"Window size must be odd, got {window_size}.")

    # Get log folder from config
    config = LoggingConfiguration(
        pow_type=pow_type,
        cluster_type=cluster_type, 
        dataset_type=dataset_type,
        task=task,
        outputs=args.outputs,
        strategy=args.strategy,
        extra_log_folder=args.extra_log_folder,
        simulator_type=simulator_type,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        batch_size=args.batch_size,
        active_learning=(args.mode == 'al_cl'),
        al_method=args.al_method,
        al_batch_size=args.al_batch_size,
        al_max_batch_size=args.al_max_batch_size,
        al_full_first_set=args.full_first_set,
        al_reload_weights=args.reload_weights,
        al_downsampling_factor=args.downsampling,
    )
    log_folder = config.get_log_folder(count=args.count, task_id=args.task_id, suffix=True)

    col_names = [
        "training_exp", "epoch", "lr",
        "Loss_Epoch", "Time_Epoch",
        "RelativeDistance_Epoch", "R2Score_Epoch",
        "Forgetting_Exp", "BWT_Exp"
    ]

    # Read training and validation loss data
    train_df = pd.read_csv(os.path.join(log_folder, "training_results_epoch.csv"))
    val_df = pd.read_csv(
        os.path.join(log_folder, "eval_results_epoch.csv"),
        header=None, names=col_names, skiprows=1
    )

    # Get number of experiences
    #num_experiences = train_df['training_exp'].nunique()
    start_exp, end_exp = args.start_exp, args.end_exp
    num_experiences = end_exp - start_exp + 1
    # Calculate optimal number of rows and columns for subplots
    # Find factors of num_experiences
    m = int(np.sqrt(num_experiences))
    for i in range(m, 0, -1):
        if num_experiences % i == 0:
            nrows, ncols = i, num_experiences // i
            break
    # Create subplot for each experience
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_experiences == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for exp in range(start_exp, start_exp + num_experiences):
        # Get data for this experience
        train_exp_data = train_df[train_df['training_exp'] == exp]
        val_exp_data = val_df[val_df['training_exp'] == exp]
        print(f"Experience {exp} has {len(val_exp_data)} epochs")
        exp_offset = exp - start_exp

        # Helper function to compute windowed average
        def compute_window_average(data, epochs, window_size):
            # Compute moving average of losses with window_size
            half_window = (window_size - 1) // 2
            smoothed = np.zeros(len(epochs))
            for i, epoch in enumerate(epochs):
                # Get window boundaries
                window_start = max(0, epoch - half_window)
                window_end = min(epochs[-1], epoch + half_window)
                
                # Get indices within window
                window_mask = (epochs >= window_start) & (epochs <= window_end)
                
                # Compute average over window
                smoothed[i] = np.mean(data[window_mask])
            return smoothed

        # Smooth training loss
        train_epochs = train_exp_data['epoch'].values
        train_loss = train_exp_data['Loss_Epoch'].values
        train_exp_data['Loss_Epoch_Smoothed'] = compute_window_average(train_loss, train_epochs, window_size)

        # Smooth validation loss 
        val_epochs = val_exp_data['epoch'].values
        val_loss = val_exp_data['Loss_Epoch'].values
        val_exp_data['Loss_Epoch_Smoothed'] = compute_window_average(val_loss, val_epochs, window_size)

        # Plot training loss
        axes[exp_offset].plot(train_exp_data['epoch'], train_exp_data['Loss_Epoch'], 
                      label='Training Loss', color='blue')
        
        # Plot smoothed training loss
        if window_size != 1:
            axes[exp_offset].plot(train_exp_data['epoch'], train_exp_data['Loss_Epoch_Smoothed'], 
                        label='Smoothed Training Loss', color='black')
        
        # Plot validation loss
        axes[exp_offset].plot(val_exp_data['epoch'], val_exp_data['Loss_Epoch'],
                      label='Validation Loss', color='red')

        # Plot smoothed validation loss
        if window_size != 1:
            axes[exp_offset].plot(val_exp_data['epoch'], val_exp_data['Loss_Epoch_Smoothed'], 
                        label='Smoothed Validation Loss', color='orange')
        
        axes[exp_offset].set_title(f'Experience {exp}')
        axes[exp_offset].set_xlabel('Epoch')
        axes[exp_offset].set_ylabel('Loss')
        axes[exp_offset].legend()
        axes[exp_offset].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_folder, f'training_validation_loss_task_{args.task_id}.png'))
    plt.show()
    plt.close()
