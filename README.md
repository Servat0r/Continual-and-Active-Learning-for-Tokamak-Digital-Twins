Final Thesis Project for Master Degree in Computer Science (AI) at University of Pisa, a.y. 2023-2024.

## Project Structure:
- `data` folder contains all used datasets, with `qualikiz` and `tglf` being the two datasets used for the thesis. `<dataset_name>/cleaned/<power_type>/<cluster_type>` contains reworked datasets for thesis experiments, in particular:
  - `complete_dataset.csv` is the whole dataset
  - `raw_<set_type>_data_<task>_<dataset_type>.csv`, with `set_type \in {train, eval, test}` and `dataset_type in {complete, not_null}` is the dataset BEFORE applying stratification, subsampling and pruning zero entries;
  - `final_<set_type>_data_<task>_<dataset_type>.csv` is the final dataset "ready for usage";
- experiment configurations are contained in JSON files, with the following fields:
  - `general` for generic configuration parameters
  - `dataset` for dataset type, inputs/outputs and normalization
  - `architecture`, `loss`, `optimizer`, `scheduler`, `early_stopping` are self-describing
  - `strategy` for CL strategies; may contain a list for sequentially running experiments with all different strategies
  - `start_model_saving` for saving of initial model states for reproducibility and usage of same initial conditions along different CL strategies and configurations
  - `active_learning` for Active Learning within Continual Learning Experiments
- `models` directory contains the weights of the models used for the experiments
- `plots` directory contains the plots for the thesis
- `plot_averages` is a package containing utilities for plotting average metrics of different CL and AL(CL) strategies across experiments
- `src` is the package containing main source code, with:
  - `src/configs/` defines how to parse JSON configurations, with handlers for each field
  - `src/utils/` contains core code for what is needed for experiments
  - `src/run/` contains code for experiments running and multiprocessing
  - `src/ex_post_tests.py` has utilities for testing models etc. also after experiments
- `main.py` is the main file for running experiments
- `mean_std_calculations.py` is used for computing several mean, std and time statistics for the thesis
- `plot_training.py` is a script for plotting training and validation metrics across experiences
- `repair_missing_plots.py` is a utility script with several fixes for some issues


## Command Line Arguments (`main.py`):
- `--config=<file_name>`: path to config file
- `--num_tasks=<num>`: number of concurrent runs to execute for the same configuration
- `--tasks <list of nums>`: task ids to run (mapped to models in `models` directory). Useful for running averages over many different tasks but splitting the runs at different times for avoiding using too many resources
- `--extra-log-folder=<path>`: name of the folder inside base log path (see below) in which to store experiment results. It can be overwritten in JSON config files
- `--no-redirect-stdout`: if given, standard output is NOT redirected to a separate file (this is the default option for multitask executions)
- `--write_intermediate_models`: if True, model weights at the end of each experience are saved in different files during experiments
- `--plot_single_runs`: if True, at the end of experiments adds plots of metrics for single runs (tasks)


## Logs Structure:
For each experiment, we define the following:
- `pow_type`: experiment power conditions (`highpow`, `lowpow` or `mixed`)
- `cluster_type`: experiment data cluster type (`Ip_Pin_based`, `tau_based`, `wmhd_based`, `beta_based`)
- `task`: `classification` (if there is or not turbulence related to one or more outputs) or `regression` (predict output values)
- `dataset_type`: `complete` means we use the whole dataset, while `not_null` means we use the subset for which we exclude zero values for all the outputs considered (for `qualikiz` and `tglf`, a subset of `{efe, efi, pfe, pfi}`)
- `outputs`: all output columns considered, in the format of `_.join(outputs)` (eg, `efe_efi_pfe_pfi`)
- `strategy`: Continual Learning Strategy used
- `extra_log_folder`: extra folder with further information for catalogating experiments
- final experiment directory in the form: `yyyy-mm-dd_hh-mm-ss <architecture or saved> task_<task num>`, in which the timestamp refers to experiment start
- `AL(CL)/Continual`: extra subfolder for AL(CL) experiments

Base directory path for experiment logs is then: `logs/<pow_type>/<cluster_type>/<task>/<dataset_type>/<outputs>/<strategy>/<extra_log_folder>`.
- For Continual Learning experiments:
  - for `qualikiz`, you directly have the folders for each experiment
  - for `tglf`, you have the subfolder `TGLF/` with then the same folders as `qualikiz`
- For AL(CL) experiments, you have the `AL(CL)/Continual/` subfolders: inside them, there is an extra folder of the form `Batches <al_batch_size> <al_max_batch_size> <full first set?> <reload weights?> <downsampling?>`, and then the same folder structure as for "pure CL" experiments.

Within these directories, for each experiment with `N` runs, there are `N` directories of the form `yyyy-mm-dd_hh-mm-ss <architecture or saved> task_<i>` for `i \in {0, ..., N-1}`.

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
- `<eval/test>_<mean/std>_values.csv`: mean and standard values for `eval_experience_results.csv` along all parallel runs
- plots of mean and standard results across parallel runs for Validation and Test Sets