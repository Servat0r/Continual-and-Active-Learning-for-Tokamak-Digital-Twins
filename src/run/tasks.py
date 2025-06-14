import gc
import json
import sys
import os
from time import perf_counter, process_time
from typing import Any, Optional
from datetime import datetime
import torch

from tqdm import tqdm
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import DatasetExperience
from avalanche.benchmarks.utils import DataAttribute

from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training import GenerativeReplay, JointTraining
from avalanche.training.plugins import EvaluationPlugin, FromScratchTrainingPlugin

from bmdal_reg.bmdal.feature_data import TensorFeatureData

from ..utils import *
from ..configs import *
from .utils import *
from .plots import *
from ..utils.active_learning import al_cl_strategy_converter
from ..utils.strategies.plugins import PercentageReplayPlugin


# Optional Synchronization
SYNC = True #False #

def synchronization(on: bool) -> None:
    """
    @param on: If True, enables torch.cuda.synchronize().
    """
    if on: torch.cuda.synchronize()


def downsample_experience(
        train_exp: DatasetExperience,
        downsampling_factor: int,
        seed: int = 42,
        factor_type: str = 'proportional' # proportional => length // factor; absolute => factor
) -> DatasetExperience:
    """
    Downsample a training experience by randomly selecting 1/downsampling_factor of the data,
    attempting to maintain stratification across orders of magnitude.
    
    Args:
        train_exp: The training experience to downsample
        downsampling_factor: Factor to downsample by (e.g. 2 means take half the data)
    
    Returns:
        Downsampled training experience
    """
    # Set seed
    torch.manual_seed(seed)
    
    if downsampling_factor <= 1:
        return train_exp

    # Get inputs and targets from the dataset
    dataset = train_exp.dataset._datasets[0]
    X, y = dataset.inputs, dataset.targets
    
    # Calculate orders of magnitude for inputs and outputs
    X_magnitudes = torch.log10(torch.abs(X) + 1e-10).mean(dim=1).floor()
    y_magnitudes = torch.log10(torch.abs(y) + 1e-10).mean(dim=1).floor()
    
    # Combine magnitudes to create strata
    combined_magnitudes = X_magnitudes * 10 + y_magnitudes  # Arbitrary scaling to separate X and y magnitudes
    unique_strata = torch.unique(combined_magnitudes)
    
    selected_indices = []
    
    sampled_sizes = None
    if factor_type == 'absolute':
        sampled_sizes = []
        for stratum in unique_strata:
            stratum_indices = torch.where(combined_magnitudes == stratum)[0]
            current_sample_size = int(round(len(stratum_indices) / len(combined_magnitudes) * downsampling_factor, 0))
            #stdout_debug_print(f"Preliminarily sampling {current_sample_size} items", color='green')
            sampled_sizes.append(current_sample_size)
        total = sum(sampled_sizes)
        #stdout_debug_print(f"Preliminary total = {total}", color='green')
        i = 0
        while total + i < downsampling_factor:
            sampled_sizes[i % len(sampled_sizes)] += 1
            i += 1
        #stdout_debug_print(f"Final sample sizes = {sampled_sizes}", color='green')
    
    # Sample from each stratum
    for idx, stratum in enumerate(unique_strata):
        stratum_indices = torch.where(combined_magnitudes == stratum)[0]
        if factor_type == 'proportional':
            num_to_sample = max(len(stratum_indices) // downsampling_factor, 0) # NOTE: Previously was 1
        elif factor_type == 'absolute':
            num_to_sample = sampled_sizes[idx]
        else:
            raise RuntimeError(f"Unknown factor_type = \"{factor_type}\"")
        #stdout_debug_print(f"Sampled = {num_to_sample}", color='cyan')
        
        # Randomly sample indices from this stratum
        if len(stratum_indices) > 0:
            sampled = stratum_indices[torch.randperm(len(stratum_indices))[:num_to_sample]]
            selected_indices.extend(sampled.tolist())
    
    # Convert to tensor and sort
    selected_indices = torch.tensor(selected_indices)
    selected_indices = selected_indices.sort()[0]
    
    # Create new dataset with selected indices
    X_sampled = X[selected_indices]
    y_sampled = y[selected_indices]
    
    new_dataset = CSVRegressionDataset(
        data=None,
        input_columns=[],
        output_columns=[],
        inputs=X_sampled,
        outputs=y_sampled,
        device=y.device,
        transform=dataset.transform,
        target_transform=dataset.target_transform
    )
    
    # Return new experience with downsampled dataset
    task_label = train_exp.dataset.targets_task_labels[0]
    old_data_attributes = []
    for name, attribute in train_exp.dataset._data_attributes.items():
        if name != "targets_task_labels":
            old_data_attributes.append(attribute)
    old_data_attributes.append(
        DataAttribute(name="targets_task_labels", data=len(new_dataset) * [task_label])
    )
    result_exp = DatasetExperience(
        current_experience=train_exp.current_experience,
        #origin_stream=train_exp.origin_stream,
        dataset=AvalancheDataset(
            [new_dataset], data_attributes=old_data_attributes
        )
    )
    result_exp._origin_stream = train_exp.origin_stream
    return result_exp


def task_training_loop(
        config_data: str | dict[str, Any], task_id: int,
        redirect_stdout=True, extra_log_folder='',
        write_intermediate_models=False,
        plot_single_runs=False,
) -> Optional[dict]:
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
    # General
    mode = config_parser['mode']
    train_mb_size = config_parser['train_mb_size']
    eval_mb_size = config_parser['eval_mb_size']
    train_epochs = config_parser['train_epochs']
    num_campaigns = config_parser['num_campaigns']
    dtype = config_parser['dtype']
    task = config_parser['task']
    full_first_train_set = config_parser['full_first_train_set']
    first_train_set_size = config_parser['first_train_set_size']
    # Dataset
    input_columns = config_parser['input_columns']
    output_columns = config_parser['output_columns']
    input_size = len(input_columns)
    output_size = len(output_columns)
    simulator_type = config_parser['simulator_type']
    pow_type = config_parser['pow_type']
    cluster_type = config_parser['cluster_type']
    dataset_type = config_parser['dataset_type']
    normalize_inputs = config_parser['normalize_inputs']
    normalize_outputs = config_parser['normalize_outputs']
    load_saved_final_data = config_parser['load_saved_final_data']
    downsampling_factor = config_parser['downsampling_factor']
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
    cl_strategy_from_scratch = cl_strategy_config['from_scratch']
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
    cl_strategy_active_learning_data = None
    batch_size = None
    max_batch_size = None
    reload_initial_weights = None
    al_method = None

    downsampling_dump_fp = None
    if mode == 'AL(CL)':
        cl_strategy_active_learning_data = config_parser['active_learning']
        if cl_strategy_active_learning_data is not None:
            batch_selector = cl_strategy_active_learning_data['batch_selector']
            batch_selector.set_models([model])
            batch_selector.set_device(device)
            batch_selector.close()
            batch_size = cl_strategy_active_learning_data['batch_size']
            max_batch_size = cl_strategy_active_learning_data['max_batch_size']
            reload_initial_weights = cl_strategy_active_learning_data['reload_initial_weights']
            al_method = cl_strategy_active_learning_data['al_method']

    # Prepare folders for experiments
    folder_name = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {model_type} task_{task_id}"
    output_columns_str = '_'.join(output_columns)
    hidden_size = config['architecture']['parameters']['hidden_size']
    hidden_layers = config['architecture']['parameters']['hidden_layers']
    scenario_config = ScenarioConfig(
        simulator_type=simulator_type, pow_type=pow_type, cluster_type=cluster_type,
        dataset_type=dataset_type, task=task, outputs=output_columns_str
    )
    logging_config = LoggingConfiguration(
        scenario=scenario_config, strategy=strategy_type, extra_log_folder=extra_log_folder,
        hidden_size=hidden_size, hidden_layers=hidden_layers, batch_size=train_mb_size, active_learning=False
    )
    if mode == 'AL(CL)':
        actual_downsampling = 1 / downsampling_factor if isinstance(downsampling_factor, int) else downsampling_factor
        logging_config.active_learning = True
        logging_config.al_config = ActiveLearningConfig(
            framework='bmdal',
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            standard_method=al_method,
            full_first_set=full_first_train_set,
            reload_initial_weights=reload_initial_weights,
            downsampling_factor=actual_downsampling
        )
    elif mode != 'CL':
        raise RuntimeError(f"Invalid mode \"{mode}\"")
    log_folder = f"{logging_config.get_log_folder(suffix=False)}/{folder_name}"
    os.makedirs(os.path.join(log_folder), exist_ok=True)
    stdout_file_path = os.path.join(log_folder, 'stdout.txt')
    if batch_selector is not None:
        batch_selector.debug_log_file = open(os.path.join(log_folder, 'batch_selector.log'), 'w')
        downsampling_dump_fp = open(os.path.join(log_folder, 'downsampling.log'), 'w')

    with open(stdout_file_path, 'w') as stdout_file:
        if redirect_stdout:
            sys.stdout = stdout_file # Redirect outputs to file
        print("Configuration Loaded:")
        print(f"  Device: {device}")
        for field_name, field_value in config_parser.get_config().items():
            print(f"  {field_name}: {field_value}")

        # Print Model Size
        trainables, total = get_model_size(model)
        print(
            f"Trainable Parameters = {trainables}"
            f"\nTotal Parameters = {total}"
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
        csv_file = f'data/{simulator_type}/cleaned/{pow_type}_cluster/{cluster_type}/complete_dataset.csv'
        if simulator_type == 'qualikiz':
            apply_subsampling = True
        elif simulator_type == 'tglf':
            apply_subsampling = False
        else:
            raise ValueError(f"Unknown simulator type \"{simulator_type}\"")
        benchmark = make_benchmark(
            csv_file, train_datasets, eval_datasets, test_datasets, task=task, NUM_CAMPAIGNS=num_campaigns,
            dtype=dtype, normalize_inputs=normalize_inputs, normalize_outputs=normalize_outputs,
            log_folder=log_folder, input_columns=input_columns, output_columns=output_columns,
            dataset_type=dataset_type, filter_by_geq=filters_by_geq, filter_by_leq=filters_by_leq,
            apply_subsampling=apply_subsampling, transform=cl_strategy_transform_transform,
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
        csv_logger = CustomCSVLogger(log_folder=log_folder, metrics=metrics, val_stream=eval_stream, verbose=False)
        has_interactive_logger = int(os.getenv('INTERACTIVE', '0'))
        loggers = ([InteractiveLogger()] if has_interactive_logger else []) + [csv_logger, mean_std_plugin]

        # Define the evaluation plugin with desired metrics
        eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)

        # Extra plugins
        plugins = [
            ValidationStreamPlugin(val_stream=eval_stream, debug_log_file=os.path.join(log_folder, 'val_stream.log')),
            TqdmTrainingEpochsPlugin(num_exp=num_campaigns, num_epochs=train_epochs),
        ]

        if SYNC:
            plugins.append(CUDASynchronizationPlugin(after_train_epoch=True, after_train_exp=True, after_eval_exp=True))

        if early_stopping:
            # Write debug log file for early stopping
            early_stopping.debug_log_file = open(os.path.join(log_folder, 'early_stopping.log'), 'w')
            plugins.append(early_stopping)
        if scheduler:
            plugins.append(scheduler)
        
        if cl_strategy_from_scratch:
            plugins.append(FromScratchTrainingPlugin(reset_optimizer=True))
        
        # Change CL strategy to AL(CL) one if needed
        if mode == 'AL(CL)':
            stdout_debug_print(f"Converting {cl_strategy_class} for AL(CL) training ...", color='red')
            cl_strategy_class = al_cl_strategy_converter(cl_strategy_class)
            stdout_debug_print(f"Converted to {cl_strategy_class} strategy", color='red')
        
        cl_strategy = cl_strategy_class(
            model=model, optimizer=optimizer, criterion=criterion, train_mb_size=train_mb_size,
            train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, evaluator=eval_plugin,
            plugins=plugins, **cl_strategy_parameters
        )
        
        if cl_strategy_class == DoubleLFL:
            cl_strategy.set_eval_stream(eval_stream)
        
        @time_logger(log_file=f'{log_folder}/timing.txt')
        def run(train_stream, eval_stream, cl_strategy, model, log_folder, write_intermediate_models):
            results = []
            if isinstance(cl_strategy, JointTraining):
                print(f"Starting JointTraining training experience: ")
                cl_strategy.train(train_stream)
                synchronization(SYNC)
                print(f"Starting JointTraining evaluation experience: ")
                results.append(cl_strategy.eval(eval_stream))
                synchronization(SYNC)
                print(f"Saving model after JointTraining experience: ")
                model.eval()
                torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_0.pt'))
                model.train()
            else:
                al_times_fp = None
                if mode == 'AL(CL)':
                    al_times_fp = open(os.path.join(log_folder, 'al_times.csv'), 'w')
                    print(*['training_exp', 'num_selected_items', 'time'], file=al_times_fp, sep=',', end='\n', flush=True)
                for (idx, train_exp), _ in zip(enumerate(train_stream), eval_stream):
                    stdout_debug_print(f"Starting training experience {idx}: ", color='green')
                    stdout_debug_print(f"Task Labels: {train_exp.dataset.targets_task_labels[0]}", color='green')
                    index_condition = (idx > 0) if full_first_train_set else True
                    if (mode == 'CL') or \
                    (mode == 'AL(CL)' and not index_condition) or \
                    (mode == 'AL(CL)' and len(train_exp.dataset) / downsampling_factor <= max_batch_size):
                        if first_train_set_size is not None:
                            # E.g., we want 5120 examples in the first training experience
                            train_exp = downsample_experience(
                                train_exp, first_train_set_size, factor_type='absolute'
                            )
                        # Training Cycle
                        cl_strategy.train(train_exp)
                        synchronization(SYNC)
                        if mode == 'AL(CL)':
                            batch_selector.set_new_train_exp(train_exp, index=idx)
                        stdout_debug_print(f"Starting testing experience {idx}: ", color='green')
                        results.append(cl_strategy.eval(eval_stream))
                        synchronization(SYNC)
                        # NOTE NEW, to see what happens with test_stream also!
                        csv_logger.set_test_stream_type()
                        cl_strategy.eval(test_stream)
                        synchronization(SYNC)
                        csv_logger.set_val_stream_type()
                    elif mode == 'AL(CL)': # Active Learning
                        cl_strategy.start_active_learning_cycle()
                        # Downsampling of the full dataset
                        orig_str = f"Original training exp {idx} has {len(train_exp.dataset)} items"
                        stdout_debug_print(orig_str, color='purple')
                        print(orig_str, file=downsampling_dump_fp)
                        train_exp = downsample_experience(train_exp, downsampling_factor)
                        new_str = f"Downsampled training exp {idx} has {len(train_exp.dataset)} items"
                        stdout_debug_print(new_str, color='purple')
                        print(new_str, file=downsampling_dump_fp)
                        # For num_iterations select from batch and add to the train_exp
                        # Store initial train_exp for pool data
                        initial_train_exp = train_exp
                        # Save initial weights
                        initial_weights = None
                        if reload_initial_weights:
                            # Save initial weights before AL training cycle
                            initial_weights = {
                                name: param.clone().detach() 
                                for name, param in model.state_dict().items()
                            }
                        num_selected_items = 0 # How many items we have selected with batch_selector
                        #csv_logger.suspend() # Deactivate CSVLogger to avoid printing out the results inside the AL cycle
                        # pool_train_exp = train_exp that contains all remaining pool data
                        # sel_train_exp = train_exp that contains all selected data
                        pool_train_exp = train_exp # todo copy?
                        pool_avalanche_dataset = pool_train_exp.dataset
                        pool_raw_dataset: CSVRegressionDataset = pool_avalanche_dataset._datasets[0]
                        sel_train_exp = None
                        current_experience = pool_train_exp.current_experience
                        al_queue = tqdm(max_batch_size // batch_size, desc="Active Learning Training")
                        while num_selected_items < max_batch_size:
                            # <------
                            al_iteration_time = perf_counter()
                            # First iteration
                            pool_inputs, pool_targets = pool_raw_dataset.get_raw_data()
                            pool_inputs_transformed = pool_raw_dataset.transform(pool_inputs)
                            pool_data = TensorFeatureData(pool_inputs_transformed)
                            # Select batch using batch_selector
                            sampled_idxs = batch_selector(pool_data, pool_avalanche_dataset)
                            sampled_idxs = sampled_idxs.to('cpu')
                            num_selected_items += batch_size
                            # Add selected samples to AL training set
                            selected_inputs = pool_inputs[sampled_idxs[:batch_size]]
                            selected_targets = pool_targets[sampled_idxs[:batch_size]]
                            # Remove selected samples from pool dataset
                            mask = torch.ones(len(pool_avalanche_dataset), dtype=torch.bool)
                            mask[sampled_idxs[:batch_size]] = False
                            pool_inputs, pool_targets = pool_inputs[mask], pool_targets[mask]
                            pool_raw_dataset.set_raw_data(pool_inputs, pool_targets)
                            task_label = pool_avalanche_dataset.targets_task_labels[0]
                            old_data_attributes = []
                            for name, attribute in pool_avalanche_dataset._data_attributes.items():
                                if name != "targets_task_labels":
                                    old_data_attributes.append(attribute)
                            pool_avalanche_dataset = AvalancheDataset(
                                [pool_raw_dataset],
                                data_attributes=old_data_attributes + [
                                    DataAttribute(
                                        name="targets_task_labels",
                                        data=len(pool_raw_dataset) * [task_label]
                                    )
                                ]
                            )
                            # NOTE: In-place modification!
                            pool_train_exp._dataset = pool_avalanche_dataset
                            if sel_train_exp is None:
                                sel_train_raw_dataset = CSVRegressionDataset(
                                    data=None, input_columns=[], output_columns=[],
                                    inputs=selected_inputs, outputs=selected_targets,
                                    device=pool_raw_dataset.device,
                                    transform=pool_raw_dataset.transform,
                                    target_transform=pool_raw_dataset.target_transform
                                )
                                sel_train_avalanche_dataset = AvalancheDataset(
                                    [sel_train_raw_dataset],
                                    data_attributes=old_data_attributes + [
                                        DataAttribute(
                                            name="targets_task_labels",
                                            data=len(sel_train_raw_dataset) * [task_label]
                                        )
                                    ]
                                )
                                sel_train_exp = DatasetExperience(
                                    dataset=sel_train_avalanche_dataset,
                                    current_experience=current_experience
                                )
                                sel_train_exp._origin_stream = pool_train_exp.origin_stream
                                # NOTE: In-place modification!
                                #sel_train_exp._dataset = sel_train_avalanche_dataset
                            else:
                                sel_train_raw_dataset: CSVRegressionDataset = sel_train_exp.dataset._datasets[0]
                                sel_train_inputs, sel_train_targets = sel_train_raw_dataset.get_raw_data()
                                sel_train_inputs = torch.cat([sel_train_inputs, selected_inputs])
                                sel_train_targets = torch.cat([sel_train_targets, selected_targets])
                                sel_train_raw_dataset = CSVRegressionDataset(
                                    data=None, input_columns=[], output_columns=[], device=sel_train_raw_dataset.device,
                                    inputs=sel_train_inputs, outputs=sel_train_targets,
                                    transform=sel_train_raw_dataset.transform,
                                    target_transform=sel_train_raw_dataset.target_transform
                                )
                                sel_train_avalanche_dataset = AvalancheDataset(
                                    [sel_train_raw_dataset],
                                    data_attributes=old_data_attributes + [
                                        DataAttribute(
                                            name="targets_task_labels",
                                            data=len(sel_train_raw_dataset) * [task_label]
                                        )
                                    ]
                                )
                                # NOTE: In-place modification!
                                sel_train_exp._dataset = sel_train_avalanche_dataset
                            if not batch_selector.replace_train_exp(sel_train_exp, index=current_experience):
                                batch_selector.set_new_train_exp(sel_train_exp, index=current_experience)
                            # Train on current AL dataset
                            # Todo THIS WILL MAKE REPLAY BE CALLED MANY TIMES
                            sel_train_exp._origin_stream = train_exp.origin_stream
                            if num_selected_items >= max_batch_size:
                                cl_strategy.stop_active_learning_cycle() # We are ready for a full training experience
                                csv_logger.resume() # Reactivate CSVLogger
                                # Restore initial model weights if specified
                                if reload_initial_weights:
                                    model.load_state_dict(initial_weights)
                                log_str = f"Train Exp length (after Batch Selection) = {len(train_exp.dataset)}"
                                stdout_debug_print(log_str, color='purple')
                                train_exp = sel_train_exp
                            al_queue.update(1)
                            # <------
                            al_iteration_time = perf_counter() - al_iteration_time
                            print(*[idx, num_selected_items, round(al_iteration_time, 4)], file=al_times_fp, sep=',', end='\n', flush=True)
                            cl_strategy.train(sel_train_exp)
                            synchronization(SYNC)
                        results.append(cl_strategy.eval(eval_stream))
                        synchronization(SYNC)
                        csv_logger.set_test_stream_type()
                        cl_strategy.eval(test_stream)
                        synchronization(SYNC)
                        csv_logger.set_val_stream_type()
                # Save models after each experience
                    if write_intermediate_models:
                        stdout_debug_print(f"Saving model after experience {idx}: ", color='red')
                        model.eval()
                        torch.save(model.state_dict(), os.path.join(log_folder, f'model_after_exp_{idx}.pt'))
                        model.train()
                if al_times_fp is not None:
                    al_times_fp.close()
            if batch_selector is not None:
                batch_selector.close()
            return results

        gc.collect()

        debug_print("[red]Starting ...[/red]", file=STDOUT)
        try:
            results = run(train_stream, eval_stream, cl_strategy, model, log_folder, write_intermediate_models)
            with open(os.path.join(log_folder, 'results.json'), 'w') as fp:
                json.dump(results, fp, indent=4)

            # Finally close the logger and evaluate on test stream
            csv_logger.close()
            final_test_results = cl_strategy.eval(test_stream)
            synchronization(SYNC)
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
            # Close debug log file for early stopping
            if early_stopping.debug_log_file is not None:
                early_stopping.debug_log_file.close()
            if batch_selector is not None:
                batch_selector.close()
            if downsampling_dump_fp is not None:
                downsampling_dump_fp.close()


__all__ = ['task_training_loop']