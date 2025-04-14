import os
from rich import print

from .common import *


if __name__ == '__main__':
    # Modes:
    # cl: default CL strategies, compares multiple CL strategies together
    # al_cl: AL(CL), choose one CL strategy and compares multiple AL methods over it
    parser = common_parser_build(with_other_params=False)
    parser.add_argument('--batch_size', nargs='+', type=int, default=[4096])
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[1024])
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[2])
    parser.add_argument('--strategy_name', type=str, default='Naive')
    parser.add_argument('--strategy_label', type=str, default='Naive')
    parser.add_argument('--extra_log_folder', type=str, default='Base')
    args = parser.parse_args()
    is_active_learning = False

    pow_type, cluster_type, task, dataset_type, simulator_type, outputs = \
        args.pow_type, args.cluster_type, args.task, args.dataset_type, args.simulator_type, args.outputs
    logging_config = LoggingConfiguration(
        pow_type=pow_type, cluster_type=cluster_type, dataset_type=dataset_type,
        task=task, outputs=outputs, simulator_type=simulator_type, hidden_size=1024,
        hidden_layers=2, batch_size=4096, active_learning=is_active_learning,
        al_batch_size=args.al_batch_size, al_max_batch_size=args.al_max_batch_size,
        al_method=args.al_method, al_full_first_set=args.full_first_set,
        al_reload_weights=args.reload_weights, al_downsampling_factor=args.downsampling
    )

    batch_sizes, hidden_sizes, hidden_layers = args.batch_size, args.hidden_size, args.hidden_layers
    strategy_name, strategy_label, extra_log_folder = args.strategy_name, args.strategy_label, args.extra_log_folder
    internal_metric_name, count, include_future_experiences, set_type = \
        args.internal_metric_name, args.count, args.include_future_experiences, args.set_type
    plot_metric_name = internal_metric_name[:-len('_Exp')]


    which = 0
    if len(batch_sizes) > 1:
        which = 1
    elif len(hidden_sizes) > 1:
        which = 2
    elif len(hidden_layers) > 1:
        which = 3
    else:
        raise RuntimeError(f"You did not provide any parameter with more than one value")
    
    if which == 1:
        hidden_sizes = hidden_sizes[0]
        hidden_layers = hidden_layers[0]
        logging_config.hidden_size = hidden_sizes
        logging_config.hidden_layers = hidden_layers
        param_name = 'batch_size'
        param_values = batch_sizes
        length = len(batch_sizes)
    elif which == 2:
        batch_sizes = batch_sizes[0]
        hidden_layers = hidden_layers[0]
        logging_config.batch_size = batch_sizes
        logging_config.hidden_layers = hidden_layers
        param_name = 'hidden_size'
        param_values = hidden_sizes
        length = len(hidden_sizes)
    else:
        batch_sizes = batch_sizes[0]
        hidden_sizes = hidden_sizes[0]
        logging_config.batch_size = batch_sizes
        logging_config.hidden_size = hidden_sizes
        param_name = 'hidden_layers'
        param_values = hidden_layers
        length = len(hidden_layers)
    
    savefolder, savepath = get_savepath(
        args, include_future_experiences, plot_metric_name, is_active_learning,
        f'{param_name.replace('_', ' ').capitalize()} Config Comparisons', '',
        strategies=[strategy_name]
    )
    os.makedirs(savefolder, exist_ok=True)

    colors_list = [(colors[i], linestyles[i]) for i in range(length)]
    mean_std_params_plots_wrapper(
        logging_config, param_name, param_values, strategy_name, strategy_label,
        extra_log_folder, f"{set_type}_mean_values.csv", f"{set_type}_std_values.csv",
        internal_metric_name, plot_metric_name, count, save=True, savepath=savepath,
        show=True, grid=True, legend=True, colors_and_linestyle_list=colors_list,
        include_future_experiences=include_future_experiences
    )
