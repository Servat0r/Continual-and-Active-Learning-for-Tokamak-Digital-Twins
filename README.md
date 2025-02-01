Final Thesis Project for Master Degree in Computer Science (AI) at University of Pisa, a.y. 2023-2024.

## Project Structure:
- `data` folder contains all used datasets, with `qualikiz` being the original dataset. `qualikiz/cleaned/<power_type>/<cluster_type>` contains reworked datasets for CL experiment, with `final_<type>_data_<task>.csv` being datasets after preprocessing and directly loaded for experiments
- experiment configurations are contained in JSON files, with the following fields:
  - `general` for generic configuration parameters
  - `dataset` for dataset type, inputs/outputs and normalization
  - `architecture`, `loss`, `optimizer`, `scheduler`, `early_stopping` are self-describing
  - `strategy` for CL strategies; may contain a list for sequentially running experiments with all different strategies
  - `start_model_saving` for saving of initial model states for reproducibility and usage of same initial conditions along different CL strategies and configurations
  - `src/configs` define how to parse JSON configurations, with handlers for each field
  - `src/utils` contains core code for what is needed for experiments
  - `src/run.py` is the main file for running experiments
  - `src/ex_post_tests.py` has utilities for testing models etc. also after experiments

## Command Line Arguments:
- `--config=<file_name>`: path to config file
- `--num_tasks=<num>`: number of concurrent runs to execute for the same configuration
- `--extra-log-folder=<path>`: name of the folder inside base log path (see below) in which to store experiment results. It can be overwritten in JSON config files
- `--no-redirect-stdout`: if given, standard output is NOT redirected to a separate file (this is the default option for multitask executions)

## Logs Structure:
For each experiment, we define the following:
- `pow_type`: experiment power conditions (`highpow` or `lowpow`)
- `cluster_type`: experiment data cluster type (`Ip_Pin_based`, `tau_based`, `pca_based`)
- `task`: `classification` (if there is or not turbulence related to one or more outputs) or `regression` (predict output values)
- `dataset_type`: `complete` means we use the whole dataset, while `not_null` means we use the subset for which we exclude zero values for all the outputs considered (for `qualikiz` and `tglf`, a subset of `{efe, efi, pfe, pfi}`)
- `outputs`: all output columns considered, in the format of `_.join(outputs)` (eg, `efe_efi_pfe_pfi`)
- `strategy`: Continual Learning Strategy used
- `extra_log_folder`: extra folder with further information for catalogating experiments
- final experiment directory in the form: `yyyy-mm-dd_hh-mm-ss <architecture or saved> task_<task num>`, in which the timestamp refers to experiment start

Base directory path for experiment logs is then: `logs/<pow_type>/<cluster_type>/<task>/<dataset_type>/<outputs>/<strategy>/<extra_log_folder>`.
Within this directory, for each experiment with `N` runs, there are `N` directories of the form `yyyy-mm-dd_hh-mm-ss <architecture or saved> task_<i>` for `i \in {0, ..., N-1}`.

Within each directory, you can find the following:
- `config.json`: experiment config
- `<train/eval/test>_results_<epoch/experience/stream>.csv`: metric results on train/validation/test stream grouped by epoch/experience/stream
- `input_<mean/std>.pt`: tensor data with mean and std used for input normalization (useful for example for subsequent experiences)
- `model_after_exp_<i>.pt`: model state dict after training experience `i`
- `model_size.txt`: total number of parameters / trainable parameters of the model
- `stdout.txt`: if stdout is redirected (by default), file that contains redirected stdout dump (useful for debugging purposes)
- `timing.txt`: time record of the whole experimentà
- plots of most important metrics with self-explaining names.

Moreover, in the `task_0` directory, you can find also:
- `<mean/std>_values.csv`: mean and standard values for `eval_experience_results.csv` along all parallel runs
- plots of mean and standard results across parallel runs