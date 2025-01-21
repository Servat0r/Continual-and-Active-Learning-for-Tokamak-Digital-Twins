import gc
import json
import sys
import os
from typing import Any
from datetime import datetime
import torch

from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training import GenerativeReplay, JointTraining, Replay, Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin

from ..utils import *
from ..configs import *
from .utils import *
from .plots import *
from ..utils.buffers import ExperienceBalancedActiveLearningBuffer
from ..utils.strategies.plugins import PercentageReplayPlugin


def task_training_loop(
        config_data: str | dict[str, Any], task_id: int,
        redirect_stdout=True, extra_log_folder='',
        write_intermediate_models=False,
        plot_single_runs=False,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config_parser = ConfigParser(config_data, task_id=task_id)
    config_parser.load_config()

    # Relevant config names
    config = config_parser.get_config()
    model_type = config['architecture']['name']
    optimizer_type = config['optimizer']['name']
    loss_type = config['loss']['name']
    strategy_type = config['strategy']['name']

    # Config Processing
    config_parser.process_config()
    debug_print(config_parser.get_config())
    # General
    mode = config_parser['mode']
    train_mb_size = config_parser['train_mb_size']
    eval_mb_size = config_parser['eval_mb_size']
    train_epochs = config_parser['train_epochs']
    num_campaigns = config_parser['num_campaigns']
    dtype = config_parser['dtype']
    task = config_parser['task']
    # Dataset
    input_columns = config_parser['input_columns']
    output_columns = config_parser['output_columns']
    input_size = len(input_columns)
    output_size = len(output_columns)
    pow_type = config_parser['pow_type']
    cluster_type = config_parser['cluster_type']
    dataset_type = config_parser['dataset_type']
    normalize_inputs = config_parser['normalize_inputs']
    normalize_outputs = config_parser['normalize_outputs']
    load_saved_final_data = config_parser['load_saved_final_data']
    # Architecture
    model = config_parser['architecture']
    initialize_weights_low(model, scale=1e-2)
    # Loss
    criterion = config_parser['loss']
    # Optimizer
    optimizer_config = config_parser['optimizer']
    optimizer_class = optimizer_config['class']
    optimizer_parameters = optimizer_config['parameters']
    optimizer = optimizer_class(model.parameters(), **optimizer_parameters)
    # Early Stopping
    early_stopping = config_parser.get_config().get('early_stopping', None)
    # Scheduler
    scheduler_config = config_parser.get_config().get('scheduler', None)
    scheduler = make_scheduler(scheduler_config, optimizer=optimizer)
    # CL Strategy Config
    cl_strategy_config = config_parser['strategy']
    cl_strategy_class = cl_strategy_config['class']
    cl_strategy_parameters = cl_strategy_config['parameters']
    extra_log_folder = cl_strategy_config.get('extra_log_folder', extra_log_folder)
    # Filters
    filters_by_geq = None
    filters_by_leq = None
    filters = config_parser.get_config().get('filters', None)
    if filters:
        filters_by_geq = filters.get('by_geq', None)
        filters_by_leq = filters.get('by_leq', None)

    # Transforms
    cl_strategy_transform = config_parser.get_config().get('transform', None)
    cl_strategy_transform_transform = None
    cl_strategy_transform_preprocess_ytrue = None
    cl_strategy_transform_preprocess_ypred = None
    if cl_strategy_transform:
        cl_strategy_transform_transform = cl_strategy_transform['transform']
        cl_strategy_transform_preprocess_ytrue = cl_strategy_transform['preprocess_ytrue']
        cl_strategy_transform_preprocess_ypred = cl_strategy_transform['preprocess_ypred']

    # Target Transforms
    cl_strategy_target_transform = config_parser.get_config().get('target_transform', None)
    cl_strategy_target_transform_transform = None
    cl_strategy_target_transform_preprocess_ytrue = None
    cl_strategy_target_transform_preprocess_ypred = None
    if cl_strategy_target_transform:
        cl_strategy_target_transform_transform = cl_strategy_target_transform['target_transform']
        cl_strategy_target_transform_preprocess_ytrue = cl_strategy_target_transform['preprocess_ytrue']
        cl_strategy_target_transform_preprocess_ypred = cl_strategy_target_transform['preprocess_ypred']

    # Active Learning
    batch_selector = None
    if mode in ['CL(AL)', 'AL(CL)']:
        cl_strategy_active_learning_data = config_parser['active_learning']
        print(cl_strategy_active_learning_data)
        if cl_strategy_active_learning_data is not None:
            batch_selector = cl_strategy_active_learning_data['batch_selector']
            batch_selector.set_models([model])
            batch_selector.set_device(device)

    # Prepare folders for experiments
    folder_name = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {model_type} task_{task_id}"
    output_columns_str = '_'.join(output_columns)
    log_folder = os.path.join(
        'logs', pow_type, cluster_type, task, dataset_type,
        output_columns_str, strategy_type, extra_log_folder, folder_name,
    )
    os.makedirs(os.path.join(log_folder), exist_ok=True)
    stdout_file_path = os.path.join(log_folder, 'stdout.txt')

    with open(stdout_file_path, 'w') as stdout_file:
        if redirect_stdout:
            sys.stdout = stdout_file # Redirect outputs to file
        print("[green]Configuration Loaded[/green]:")
        print(f"  Device: [cyan]{device}[/cyan]")
        for field_name, field_value in config_parser.get_config().items():
            print(f"  {field_name}: [cyan]{field_value}[/cyan]")

        # Print Model Size
        trainables, total = get_model_size(model)
        print(
            f"[green]Trainable Parameters[/green] = [red]{trainables}[/red]"
            f"\n[green]Total Parameters[/green] = [red]{total}[/red]"
        )

        # Saving model before usage
        start_model_saving_data = config_parser['start_model_saving']
        if start_model_saving_data:
            saved_model_folder = start_model_saving_data['saved_model_folder']
            saved_model_name = start_model_saving_data['saved_model_name']
            os.makedirs(saved_model_folder, exist_ok=True)
            with open(f'{saved_model_folder}/{saved_model_name}.json', 'w') as fp:
                json.dump(config['architecture'], fp, indent=4)
            torch.save(model.state_dict(), f'{saved_model_folder}/{saved_model_name}.pt')

        # Print model size to experiment directory
        with open(f'{log_folder}/model_size.txt', 'w') as fp:
            print(
                f"Trainable Parameters = {trainables}"
                f"\nTotal Parameters = {total}",
                file=fp
            )

        train_datasets = []
        eval_datasets = []
        test_datasets = []
        csv_file = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}/complete_dataset.csv'
        print(
            f"Input columns = {input_columns}"
            f"\nOutput columns = {output_columns}"
        )
        benchmark = make_benchmark(
            csv_file, train_datasets, eval_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns,
            dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
            log_folder=log_folder, input_columns=input_columns, output_columns=output_columns,
            dataset_type=dataset_type, filter_by_geq=filters_by_geq, filter_by_leq=filters_by_leq,
            apply_subsampling=True, transform=cl_strategy_transform_transform,
            target_transform=cl_strategy_target_transform_transform,
            load_saved_final_data=load_saved_final_data,
        )

        train_stream = benchmark.train_stream
        eval_stream = benchmark.eval_stream
        test_stream = benchmark.test_stream

        if cl_strategy_class != GenerativeReplay:
            with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
                json.dump(config, fp, indent=4)

        # Get and transform metrics
        metrics = get_metrics(loss_type)
        if cl_strategy_target_transform:
            metrics = preprocessed_metrics(
                metrics, preprocess_ytrue=cl_strategy_target_transform_preprocess_ytrue,
                preprocess_ypred=cl_strategy_target_transform_preprocess_ypred
            )
        metrics = loss_metrics(epoch=True, experience=True, stream=True) + metrics

        # Build logger
        mean_std_plugin = MeanStdPlugin([str(metric) for metric in metrics], num_experiences=num_campaigns)
        csv_logger = CustomCSVLogger(log_folder=log_folder, metrics=metrics, val_stream=eval_stream)
        has_interactive_logger = int(os.getenv('INTERACTIVE', '0'))
        loggers = ([InteractiveLogger()] if has_interactive_logger else []) + [csv_logger, mean_std_plugin]

        # Define the evaluation plugin with desired metrics
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)

        # Extra plugins
        plugins = [
            ValidationStreamPlugin(val_stream=eval_stream),
            TqdmTrainingEpochsPlugin(num_exp=num_campaigns, num_epochs=train_epochs),
        ]

        if early_stopping:
            plugins.append(early_stopping)
        if scheduler:
            plugins.append(scheduler)

        print(f"CL Strategy Class: {cl_strategy_class}")
        if mode == 'CL(AL)':
            if cl_strategy_class == Replay:
                cl_strategy_class = Naive
                mem_size = cl_strategy_parameters.pop('mem_size')
                replay_plugin = ReplayPlugin(
                    mem_size=mem_size,
                    storage_policy=ExperienceBalancedActiveLearningBuffer(
                        max_size=mem_size, batch_selector=batch_selector, device=device
                    )
                )
                plugins.append(replay_plugin)
            elif cl_strategy_class == PercentageReplay:
                cl_strategy_class = Naive
                mem_percentage = cl_strategy_parameters.pop('mem_percentage')
                replay_plugin = PercentageReplayPlugin(
                    mem_percentage=mem_percentage,
                    storage_policy=ExperienceBalancedActiveLearningBuffer(
                        max_size=50_000, batch_selector=batch_selector, device=device # TODO Fix this "magic number"
                    )
                )
                plugins.append(replay_plugin)

        cl_strategy = cl_strategy_class(
            model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
            train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins, **cl_strategy_parameters
        )

        @time_logger(log_file=f'{log_folder}/timing.txt')
        def run(train_stream, eval_stream, cl_strategy, model, log_folder, write_intermediate_models):
            results = []
            if isinstance(cl_strategy, JointTraining):
                print(f"Starting [red]JointTraining[/red]training experience: ")
                cl_strategy.train(train_stream)
                print(f"Starting [red]JointTraining[/red] evaluation experience: ")
                results.append(cl_strategy.eval(eval_stream))
                print(f"Saving model after [red]JointTraining[/red] experience: ")
                model.eval()
                torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_0.pt'))
                model.train()
            else:
                current_metrics = None
                for (idx, train_exp), eval_exp in zip(enumerate(train_stream), eval_stream):
                    # Active Learning
                    if batch_selector is not None:
                        batch_selector.set_train_exp(train_exp)
                    if (idx > 0) and (batch_selector is not None) and (mode == 'AL(CL)'):
                        ... # TODO Complete later!
                    print(f"Starting training experience [red]{idx}[/red]: ")
                    # Early Stopping stuff
                    if not early_stopping.use_validation_plugin:
                        early_stopping.update(cl_strategy, current_metrics)
                    # Training Cycle
                    cl_strategy.train(train_exp)
                    print(f"Starting testing experience [red]{idx}[/red]: ")
                    results.append(cl_strategy.eval(eval_stream))
                    # Save models after each experience
                    if write_intermediate_models:
                        print(f"Saving model after experience [red]{idx}[/red]: ")
                        model.eval()
                        torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_{idx}.pt'))
                        model.train()
            return results

        # garbage collect before running
        print("Garbage collecting ...")
        gc.collect()

        # train and test loop over the stream of experiences
        print("Starting ...")
        try:
            results = run(train_stream, eval_stream, cl_strategy, model, log_folder, write_intermediate_models)
            with open(os.path.join(log_folder, 'results.json'), 'w') as fp:
                json.dump(results, fp, indent=4)

            # Finally close the logger and evaluate on test stream
            csv_logger.set_test_stream_type()
            final_test_results = cl_strategy.eval(test_stream)
            csv_logger.close()
            # Filter by results on test_stream
            final_test_results = {
                k: v for k, v in final_test_results.items() if "test_stream" in k
            }
            final_test_results = process_test_results(final_test_results)
            with open(os.path.join(log_folder, 'final_test_results.json'), 'w') as fp:
                json.dump(final_test_results, fp, indent=4)
            model.eval()
            # Save model for future usage
            torch.save(model.state_dict(), os.path.join(log_folder, 'model.pt'))
            # Plots
            metric_list = get_metric_names_list(task)
            title_list = get_title_names_list(task)
            ylabel_list = get_ylabel_names_list(task)
            if plot_single_runs:
                evaluation_experiences_plots(log_folder, metric_list, title_list, ylabel_list)
            mean_std_plugin.dump_results(os.path.join(log_folder, "mean_std_metric_dump.csv"))
            return {
                'result': True, 'log_folder': log_folder, 'task': task,
                'is_joint_training': (cl_strategy_class == JointTraining)
            }
        except Exception as ex:
            raise ex
        finally:
            # Reset stdout
            if redirect_stdout:
                sys.stdout = sys.__stdout__


__all__ = ['task_training_loop']