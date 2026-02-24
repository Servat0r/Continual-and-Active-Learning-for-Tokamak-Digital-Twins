For wmhd-based and beta-based clusters experiments:
- config files are in configs/mixed/not_null directory:
    - mlp_regression_QLK_CL.json is for the CL-only experiments
    - mlp_regression_QLK_AL_CL.json is for the CL + AL experiments

Config files parameters:
    - general:
        - mode, task, dtype should be left unchanged
        - train_epochs: (max) train epochs per experience (defaults to 200 per experience)
    - dataset:
        - pow_type, input_columns, output_columns, input_size, output_size, simulator_type should be left unchanged
        - dataset_type:
            I distinguished between "complete" (all data) and "not_null" (EXCLUDING rows where efe=efi=pfe=pfi=0).
            For the thesis I did everything on "not_null"
        - normalize_inputs: whether to normalize inputs for the NNs (set to true)
        - normalize_outputs: same for outputs, although it did provide any benefit (then set to false)
        - load_saved_final_data: if false, loads data from raw datasets, preprocesses them and save preprocessed files
            in the data/qualikiz/cleaned/<cluster_name> directory. First usage time should be set to false, then to true.
    - architecture:
        I have made things such that you can save and reload initial model weights (if I remember correctly, I did not set all seeds)
        - name: first time should be set to "MLP", then to "saved" (otherwise it will generate other initial model weights)
        - model_folder: subdirectory in models/ where to save model weights (models/<model folder>)
        - model_name, model_class_name: should be left unchanged
        - parameters: model parameters (for beta and wmhd, input_size and output_size are fixed to 15 and 4, while the others can change)
            NOTE: If using "saved" for name, parameters should match those of saved model weights
    - loss, optimizer, scheduler: self-explicative, can be changed or not
    - strategy: list of strategy configurations
        - extra_log_folder: save path for experiments is:
            logs/mixed (or highpow or lowpow)/wmhd_based (or beta_based)/
                regression (or classification)/not_null (or complete)/efe_efi_pfe_pfi/
                <strategy name>/<extra_log_folder>
            Within each of those folders, there is a single different folder for each task, named according to the format:
            yyyy-mm-dd_hh-mm-ss <architecture.name> task_<task id, eg from 0 to 3>
        - ignore: if true, ignores the current strategy configuration item for the current run. Set to false to run (also) that strategy
        - parameters: parameters for strategy
    - early_stopping: self-explicative, can be changed or not
    - validation_stream: leave unchanged
    - start_model_saving:
        - save_model: first time set to true for saving initial model weights,
            then set to false (if you want to re-use previously-saved initial model weights)
        - saved_model_folder: folder name in which to save initial model weights (should match with architecture.model_folder to load them in future runs)
        - saved_model_name, add_timestamp: leave unchanged


FOR RUNNING EXPERIMENTS, use:
    python main.py --config=<path to config file> --num_tasks=<num of parallel runs>

    eg

    python main.py --config=configs/mixed/not_null/mlp_regression_QLK_CL.json


ATFER FIRST RUN, check that you have the data/qualikiz/cleaned/mixed_cluster (or highpow_cluster or lowpow_cluster)/wmhd_based (or beta_based)
directory with the following files:
    - complete_dataset.csv
    - final_train_data_regression_not_null.csv (if regression task + not_null dataset), and similarly for eval and test
    - raw_train_data_regression_not_null.csv (if regression task + not_null dataset), and similarly for eval and test
