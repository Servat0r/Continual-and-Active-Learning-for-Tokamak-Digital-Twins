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

if __name__ == '__main__':
    args = parser.parse_args()
    pow_type, cluster_type, dataset_type, task, simulator_type = \
        args.pow_type, args.cluster_type, args.dataset_type, args.task, args.simulator_type

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
    num_experiences = train_df['training_exp'].nunique()
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

    for exp in range(num_experiences):
        # Get data for this experience
        train_exp_data = train_df[train_df['training_exp'] == exp]
        val_exp_data = val_df[val_df['training_exp'] == exp]
        print(len(val_exp_data))

        # Plot training loss
        axes[exp].plot(train_exp_data['epoch'], train_exp_data['Loss_Epoch'], 
                      label='Training Loss', color='blue')
        
        # Plot validation loss
        axes[exp].plot(val_exp_data['epoch'], val_exp_data['Loss_Epoch'],
                      label='Validation Loss', color='red')

        axes[exp].set_title(f'Experience {exp}')
        axes[exp].set_xlabel('Epoch')
        axes[exp].set_ylabel('Loss')
        axes[exp].legend()
        axes[exp].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_folder, f'training_validation_loss_task_{args.task_id}.png'))
    plt.show()
    plt.close()
