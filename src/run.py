import json
import sys
import os
import shutil
from datetime import datetime
import argparse

from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

from joblib import Parallel, delayed

sys.path.append(os.path.dirname(__file__))  # Add src directory to sys.path

from .utils import *
from .configs import *


def make_scheduler(scheduler_config, optimizer):
    if scheduler_config:
        scheduler_class = scheduler_config['class']
        scheduler_parameters = scheduler_config['parameters']
        scheduler_metric = scheduler_config['metric']
        scheduler_first_epoch_only = scheduler_config['first_epoch_only']
        scheduler_first_exp_only = scheduler_config['first_exp_only']
        return LRSchedulerPlugin(
            scheduler_class(optimizer, **scheduler_parameters),
            metric=scheduler_metric,
            first_exp_only=scheduler_first_exp_only,
            first_epoch_only=scheduler_first_epoch_only,
        )
    else:
        return None


def get_metrics(loss_type):
    if loss_type == 'GaussianNLL':
        metrics = \
            loss_metrics(epoch=True, experience=True, stream=True) + \
            gaussian_mse_metrics(epoch=True, experience=True, stream=True) + \
            renamed_forgetting_metrics(experience=True, stream=True) + \
            renamed_bwt_metrics(experience=True, stream=True)
    else:
        metrics = \
            loss_metrics(epoch=True, experience=True, stream=True) + \
            relative_distance_metrics(epoch=True, experience=True, stream=True) + \
            r2_score_metrics(epoch=True, experience=True, stream=True) + \
            renamed_forgetting_metrics(experience=True, stream=True) + \
            renamed_bwt_metrics(experience=True, stream=True)
    return metrics


def evaluation_experiences_plots(log_folder, metric_list, title_list, ylabel_list):
    for metric, title, ylabel in zip(metric_list, title_list, ylabel_list):
        # Experiences 0-4
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, start_exp=0, end_exp=4,
            savepath=os.path.join(log_folder, f'plot_of_first_5_experiences_{metric[:-4]}.png'),
        )
        # Plot over all experiences
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, start_exp=0, end_exp=9,
            savepath=os.path.join(log_folder, f'plot_of_all_10_experiences_{metric[:-4]}.png'),
        )


def task_training_loop(config_file_path: str, task_id: int):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config_parser = ConfigParser(config_file_path, task_id=task_id)
    debug_print(config_parser.parsing_dict)
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
    # Architecture
    model = config_parser['architecture']
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

    # Prepare folders for experiments
    folder_name = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {model_type} task_{task_id}"
    log_folder = os.path.join(
        'logs', pow_type, cluster_type, task, dataset_type, strategy_type, folder_name
    )
    os.makedirs(os.path.join(log_folder), exist_ok=True)
    stdout_file_path = os.path.join(log_folder, 'stdout.txt')

    with open(stdout_file_path, 'w') as stdout_file:
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
        test_datasets = []
        csv_file = f'data/baseline/cleaned/{pow_type}_cluster/{cluster_type}/complete_dataset.csv'
        print(
            f"Input columns = {input_columns}"
            f"\nOutput columns = {output_columns}"
        )
        benchmark = make_benchmark(
            csv_file, train_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns,
            dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
            log_folder=log_folder, input_columns=input_columns, output_columns=output_columns,
            dataset_type=dataset_type, filter_by_geq=filters_by_geq, filter_by_leq=filters_by_leq,
            transform=cl_strategy_transform_transform,
            target_transform=cl_strategy_target_transform_transform,
        )

        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream

        with open(os.path.join(log_folder, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)
        metrics = get_metrics(loss_type)
        if cl_strategy_target_transform and False:
            metrics = preprocessed_metrics(
                metrics, preprocess_ytrue=cl_strategy_target_transform_preprocess_ytrue,
                preprocess_ypred=cl_strategy_target_transform_preprocess_ypred
            )
        # Build logger
        mean_std_plugin = MeanStdPlugin([str(metric) for metric in metrics], num_experiences=num_campaigns)
        csv_logger = CustomCSVLogger(log_folder=log_folder, metrics=metrics, val_stream=test_stream)
        has_interactive_logger = int(os.getenv('INTERACTIVE', '0'))
        loggers = ([InteractiveLogger()] if has_interactive_logger else []) + [csv_logger, mean_std_plugin]
        # Define the evaluation plugin with desired metrics
        if task == 'regression':
            eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)
        else:
            eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)

        # Extra plugins
        plugins = [
            ValidationStreamPlugin(val_stream=test_stream),
            TqdmTrainingEpochsPlugin(num_exp=num_campaigns, num_epochs=train_epochs),
        ]

        if early_stopping:
            plugins.append(early_stopping)
        if scheduler:
            plugins.append(scheduler)

        # Continual learning strategy
        cl_strategy = cl_strategy_class(
            model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
            train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins, **cl_strategy_parameters
        )

        @time_logger(log_file=f'{log_folder}/timing.txt')
        def run(train_stream, test_stream, cl_strategy, model, log_folder):
            results = []
            for idx, train_exp in enumerate(train_stream):
                print(f"Starting training experience [red]{idx}[/red]: ")
                cl_strategy.train(train_exp)
                print(f"Starting testing experience [red]{idx}[/red]: ")
                results.append(cl_strategy.eval(test_stream))
                print(f"Saving model after experience [red]{idx}[/red]: ")
                model.eval()
                torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_{idx}.pt'))
                model.train()
            return results

        # train and test loop over the stream of experiences
        print("[#aa0000]Starting ...[/#aa0000]")
        try:
            results = run(train_stream, test_stream, cl_strategy, model, log_folder)
            with open(os.path.join(log_folder, 'results.json'), 'w') as fp:
                json.dump(results, fp, indent=4)

            model.eval()
            # Save model for future usage
            torch.save(model.state_dict(), os.path.join(log_folder, 'model.pt'))
            # Plots
            evaluation_experiences_plots(
                log_folder, ['Loss_Exp', 'R2Score_Exp', 'RelativeDistance_Exp'],
                ['Loss over each experience', 'R2 Score over each experience', 'Relative Distance over each experience'],
                ['Loss', 'R2 Score', 'Relative Distance']
            )
            mean_std_plugin.dump_results(os.path.join(log_folder, "mean_std_metric_dump.csv"))
            return True
        except Exception as ex:
            debug_print("Caught Exception: ", ex)
            try:
                shutil.rmtree(log_folder)
            except:
                debug_print(f"[red]Failed to remove {log_folder}[/red]")
            finally:
                raise ex


def main():
    # Config
    if len(sys.argv) < 2:
        print(f"[red]Usage[/red]: [cyan](python) {sys.argv[0]} configuration_file_name.json [OPTIONS] [/cyan]")
        print(f"[cyan]Options: [/cyan]")
        print(f"[cyan]--num_tasks=<int> [/cyan] (parallel execution on multiple model instances)")
        print(f"[cyan]-h[/cyan] or [cyan]--help[/cyan] (help messages)")
        sys.exit(1)

    # Argument Parsing
    cmd_arg_parser = argparse.ArgumentParser(
        description="Parse command-line options with specific parameters."
    )
    cmd_arg_parser.add_argument(
        '--config',
        type=str,
        help="JSON configuration file to load."
    )
    cmd_arg_parser.add_argument(
        '--num_tasks',
        type=int,
        help="Number of tasks to execute (e.g., --num_tasks=16). If <= 0, it defaults to os.cpu_count().",
    )

    # Parse arguments
    cmd_args = cmd_arg_parser.parse_args()
    config_file_path = cmd_args.config
    print(f"Config file path: {config_file_path}")
    if cmd_args.num_tasks <= 0:
        num_jobs = os.cpu_count()
    else:
        num_jobs = cmd_args.num_tasks
    print(f"Number of jobs: {num_jobs}")
    if num_jobs > 1:
        task_ids = range(num_jobs)
        results = \
            Parallel(n_jobs=num_jobs)(
                delayed(task_training_loop)(config_file_path, task_id) for task_id in task_ids
            )
        print(results)
    else:
        results = task_training_loop(config_file_path, 0)
        print(results)


if __name__ == '__main__':
    main()
