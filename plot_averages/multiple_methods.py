# Plotting for multiple configurations of a single strategy
import os
from rich import print

from .common import *


if __name__ == '__main__':
    parser = common_parser_build(with_mode=False, with_al_method=False, with_strategies=False)
    parser.add_argument('--al_methods', nargs='+', type=str, default=[])
    parser.add_argument('--strategy', type=str, default='Cumulative')
    parser.add_argument('--extra_log_folder', type=str, default='Base')
    args = parser.parse_args()

    strategy, extra_log_folder = args.strategy, args.extra_log_folder
    al_cl_strategies_dictionary = get_al_cl_strategies_dictionary(
        args.simulator_type, args.pow_type, args.cluster_type, args.dataset_type, args.task
    )
    data = al_cl_strategies_dictionary[strategy]
    for item in data:
        if item[1] == extra_log_folder:
            stdout_debug_print(f"Selected {strategy} => ({item[0]}, {item[1]})", color='cyan')
            selected_item = item[:]
            extra_log_folder = item[1]
    
    strategies = {strategy: selected_item}
    al_methods = {
        'random_sketch_grad': al_cl_methods_dictionary['random_sketch_grad'], # Lower Baseline
        'lcmd_sketch_grad': al_cl_methods_dictionary['lcmd_sketch_grad'], # Upper Baseline
    }
    for method in args.al_methods:
        al_methods[method] = al_cl_methods_dictionary[method]
    
    outputs = args.outputs

    simulator_prefix = simulator_prefixes[args.simulator_type]
    if args.internal_metric_name.endswith('_Exp'):
        plot_metric_name = args.internal_metric_name[:-4]
    else:
        plot_metric_name = args.internal_metric_name[:]
    
    index = 0
    pow_type, cluster_type, task, dataset_type, simulator_type = \
        args.pow_type, args.cluster_type, args.task, args.dataset_type, args.simulator_type
    logging_config = LoggingConfiguration(
        pow_type=pow_type, cluster_type=cluster_type, dataset_type=dataset_type, task=task,
        outputs=outputs, simulator_type=simulator_type, hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers, batch_size=args.batch_size, active_learning=True,
        al_batch_size=args.al_batch_size, al_max_batch_size=args.al_max_batch_size, # al_method NOT specified
        al_full_first_set=args.full_first_set, al_reload_weights=args.reload_weights,
        al_downsampling_factor=args.downsampling
    )
    
    strategies_dict = {}
    al_methods_dict = {}
    colors_dict = {}
    savefolder = f"plots/AL(CL)/Multiple Method Comparisons/{outputs}/{simulator_type}/{pow_type}/{cluster_type}/{outputs}/{plot_metric_name}"
    savepath = f"{savefolder}/{', '.join(strategies)} {args.hidden_size}-{args.hidden_layers} on Eval Set.png"

    os.makedirs(savefolder, exist_ok=True)
    for strategy, (proxy, extra_log_folder) in strategies.items():
        logging_config.strategy = strategy
        logging_config.extra_log_folder = extra_log_folder
        print(logging_config)
        for al_method, al_method_proxy in al_methods.items():
            logging_config.al_method = al_method
            try:
                folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=True)
                print(
                    f"[red]Strategy: {strategy}, AL Method: {al_method}, " + \
                    f"Proxy: {proxy}, AL Proxy: {al_method_proxy}, Folder: {extra_log_folder}[/red]"
                )
                final_proxy_name = f"{al_method_proxy}"
                al_methods_dict[final_proxy_name] = (al_method, extra_log_folder)
                colors_dict[final_proxy_name] = (colors[index], linestyles[index])
                index += 1
            except FileNotFoundError as ex:
                folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=False)
                stdout_debug_print(f"{folder_path}: {ex.args[1:]}", color='green')
    mean_std_al_plots_wrapper(
        logging_config, al_methods_dict, internal_metric_name=args.internal_metric_name,
        plot_metric_name=plot_metric_name, count=-1, show=True, save=True, savepath=savepath,
        grid=True, legend=True, colors_and_linestyle_dict=colors_dict,
        pure_cl_strategy=strategy, pure_cl_extra_log_folder=extra_log_folder
    )
