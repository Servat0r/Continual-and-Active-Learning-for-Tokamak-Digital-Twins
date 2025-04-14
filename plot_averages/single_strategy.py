# Plotting for multiple configurations of a single strategy
import os
import copy
from rich import print

from .common import *


if __name__ == '__main__':
    # Modes:
    # cl: default CL strategies, compares multiple CL strategies together
    # al_cl: AL(CL), choose one CL strategy and compares multiple AL methods over it
    parser = common_parser_build()
    args = parser.parse_args()
    args.show = bool(args.show)
    is_active_learning = args.mode == 'al_cl'

    cl_strategies_dictionary = get_cl_strategies_dictionary(
        args.simulator_type, args.pow_type, args.cluster_type, args.dataset_type, args.task
    )
    al_cl_strategies_dictionary = get_al_cl_strategies_dictionary(
        args.simulator_type, args.pow_type, args.cluster_type, args.dataset_type, args.task
    )
    if args.mode == 'cl':
        strategies = {}
        if not args.exclude_naive:
            strategies['Naive'] = cl_strategies_dictionary['Naive'] # Lower Baseline
        if not args.exclude_cumulative:
            strategies['Cumulative'] = cl_strategies_dictionary['Cumulative'] # Upper Baseline
        for strategy in args.strategies:
            strategies[strategy] = cl_strategies_dictionary[strategy]
        al_methods = {}
    elif is_active_learning:
        strategies = {strategy: al_cl_strategies_dictionary[strategy] for strategy in args.strategies}
        al_value = al_cl_methods_dictionary[args.al_method]
        al_methods = {
            'random_sketch_grad': al_cl_methods_dictionary['random_sketch_grad'], # Lower Baseline
            'lcmd_sketch_grad': al_cl_methods_dictionary['lcmd_sketch_grad'], # Upper Baseline
            args.al_method: al_value
        }
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    outputs = args.outputs
    include_future_experiences = args.include_future_experiences
    set_type = args.set_type

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
        hidden_layers=args.hidden_layers, batch_size=args.batch_size, active_learning=is_active_learning,
        al_batch_size=args.al_batch_size, al_max_batch_size=args.al_max_batch_size, al_method=args.al_method,
        al_full_first_set=args.full_first_set, al_reload_weights=args.reload_weights,
        al_downsampling_factor=args.downsampling
    )
    
    strategies_dict = {}
    #al_methods_dict = {}
    colors_dict = {}
    savefolder, savepath = get_savepath(
        args, include_future_experiences, plot_metric_name, is_active_learning,
        'Strategy Config Comparisons', 'Strategy and Method Config Comparisons',
        strategies=strategies
    )
    if args.savepath is not None:
        savepath = args.savepath
        savefolder = os.path.dirname(savepath)
    print(f"Savefolder = {savefolder}\nSavepath = {savepath}")
    os.makedirs(savefolder, exist_ok=True)

    logging_config_base = copy.deepcopy(logging_config)
    configs_and_dicts = []
    for strategy, strategy_data in strategies.items():
        for (proxy, extra_log_folder) in strategy_data:
            logging_config = copy.deepcopy(logging_config_base)
            logging_config.strategy = strategy
            logging_config.extra_log_folder = extra_log_folder
            al_methods_dict = {}
            if args.mode == 'cl':
                try:
                    folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=True)
                    print(f"[red]Strategy: {strategy}, Proxy: {proxy}, Folder: {extra_log_folder}[/red]")
                    strategies_dict[proxy] = (strategy, extra_log_folder)
                    colors_dict[proxy] = (colors[index], linestyles[index])
                    index += 1
                except FileNotFoundError as ex:
                    folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=False)
                    stdout_debug_print(f"{folder_path}: {ex.args[1:]}", color='green')
            else:
                for al_method, al_method_proxy in al_methods.items():
                    logging_config.al_method = al_method
                    try:
                        folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=True)
                        print(
                            f"[red]Strategy: {strategy}, AL Method: {al_method}, " + \
                            f"Proxy: {proxy}, AL Proxy: {al_method_proxy}, Folder: {extra_log_folder}[/red]"
                        )
                        final_proxy_name = f"{al_method_proxy} - {proxy}"
                        al_methods_dict[final_proxy_name] = (al_method, extra_log_folder)
                        colors_dict[final_proxy_name] = (colors[index], linestyles[index])
                        index += 1
                    except FileNotFoundError as ex:
                        folder_path = logging_config.get_log_folder(count=-1, task_id=0, suffix=False)
                        stdout_debug_print(f"{folder_path}: {ex.args[1:]}", color='green')
                configs_and_dicts.append((logging_config, al_methods_dict))
    
    if args.mode == 'cl':
        mean_std_strategy_plots_wrapper(
            logging_config, strategies_dict, internal_metric_name=args.internal_metric_name,
            mean_filename=f"{set_type}_mean_values.csv", std_filename=f"{set_type}_std_values.csv",
            plot_metric_name=plot_metric_name, show=args.show, save=True, savepath=savepath,
            grid=True, legend=True, count=-1, colors_and_linestyle_dict=colors_dict,
            include_future_experiences=include_future_experiences,
            include_std=bool(args.include_std)
        )
    else:
        mean_std_al_plots_wrapper(
            configs_and_dicts, internal_metric_name=args.internal_metric_name,
            mean_filename=f"{set_type}_mean_values.csv", std_filename=f"{set_type}_std_values.csv",
            plot_metric_name=plot_metric_name, count=-1, show=args.show, save=True, savepath=savepath,
            grid=True, legend=True, colors_and_linestyle_dict=colors_dict,
            include_future_experiences=include_future_experiences,
            include_std=bool(args.include_std)
        )
