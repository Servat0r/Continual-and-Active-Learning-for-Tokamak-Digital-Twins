import os
import pandas as pd
import numpy as np
from torch import float16, float32, float64
from datetime import datetime
from dotenv import load_dotenv
from rich import print as dbprint
from pathlib import Path


load_dotenv(
    ".env",
    override=True,
    verbose=True
)


def debug_print(*objects, sep=' ', end='\n', file=None, flush=True):
    debug = int(os.getenv("DEBUG", "0"))
    if debug:
        dbprint(
            *objects,
            sep=sep,
            end=end,
            file=file,
            flush=flush,
        )


def get_dtype_from_str(dtype: str):
    if dtype == 'float16':
        dtype = float16
    elif dtype == 'float32':
        dtype = float32
    elif dtype == 'float64':
        dtype = float64
    else:
        raise ValueError(f"Unknown dtype \"{dtype}\"")
    return dtype


def time_logger(log_file=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            print(f"[{func.__name__}] Start Time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
            result = func(*args, **kwargs)
            end_time = datetime.now()
            print(f"[{func.__name__}] End Time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
            elapsed_time = end_time - start_time
            print(f"[{func.__name__}] Elapsed Time:", elapsed_time)
            if log_file:
                with open(log_file, "a") as fp:
                    print(f"[{func.__name__}] Elapsed Time: {elapsed_time}", file=fp, flush=True)
            return result
        return wrapper
    return decorator


def extract_metric_info(metric_name: str) -> dict[str, str]:
    values = metric_name.split("/")
    results = {
        'name': values[0],
        'phase': values[1],
        'stream': values[2] if len(values) > 2 else None,
        'exp': values[3] if len(values) > 3 else None,
        'exp_number': int(values[3][3:]) if len(values) > 3 else None,
    }
    if results['name'].endswith('MB'):
        results['type'] = 'minibatch'
    elif results['name'].endswith('Epoch'):
        results['type'] = 'epoch'
    elif results['name'].endswith('Exp'):
        results['type'] = 'exp'
    elif results['name'].endswith('Stream'):
        results['type'] = 'stream'
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
    return results


def extract_metric_type(metric_name: str):
    if metric_name.endswith('MB'):
        return metric_name[:-6], 'minibatch'
    elif metric_name.endswith('Epoch'):
        return metric_name[:-6], 'epoch'
    elif metric_name.endswith('Exp'):
        return metric_name[:-4], 'exp'
    elif metric_name.endswith('Stream'):
        return metric_name[:-7], 'stream'
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


def get_means_std_over_evaluation_experiences_multiple_runs(
        file_paths_or_bufs: list[str | pd.DataFrame], mean_savepath: str, std_savepath: str
):
    dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths_or_bufs]
    columns = dfs[0].columns
    mean_df = pd.DataFrame(columns=columns)
    std_df = pd.DataFrame(columns=columns)
    for column in columns:
        values = [df[column].to_numpy(dtype=np.float32) for df in dfs]
        arr = np.round(np.vstack(values), decimals=8)
        mean_df[column] = arr.mean(axis=0)
        std_df[column] = arr.std(axis=0)
    mean_df.to_csv(mean_savepath, index=False)
    std_df.to_csv(std_savepath, index=False)
    return mean_df, std_df


def extract_metric_values_over_evaluation_experiences(
    file_paths_or_bufs: list[str | pd.DataFrame], metric: str, num_exp: int = None,
):
    dfs: list[pd.DataFrame] = [pd.read_csv(fp) if isinstance(fp, str) else fp for fp in file_paths_or_bufs]
    num_exp = num_exp if num_exp is not None else len(dfs[0]['eval_exp'].unique())
    result_dfs = []
    for df in dfs:
        result_df = pd.DataFrame()
        for exp_id in range(num_exp):
            data = df[df['eval_exp'] == exp_id][metric].to_numpy()
            result_df[exp_id] = data
        result_dfs.append(result_df)
    return result_dfs


def get_all_tasks_paths(base_path: str):
    path = Path(base_path)
    directories = [os.path.join(base_path, d.name) for d in path.iterdir() if d.is_dir()]
    # Order as task_0, task_1, task_2 etc
    directories = sorted(directories, key=lambda x: int(x[-1]))
    files = [
        os.path.join(dir_path, 'eval_results_experience.csv') for dir_path in directories
    ]
    mean_path = os.path.join(directories[0], 'mean_eval_results_experience.csv')
    std_path = os.path.join(directories[0], 'std_eval_results_experience.csv')
    return {
        'directories': directories,
        'files': files,
        'mean_path': mean_path,
        'std_path': std_path,
    }


__all__ = [
    "debug_print", "time_logger", "float16", "float32", "float64",
    "get_dtype_from_str", "extract_metric_info", "extract_metric_type",
    "get_means_std_over_evaluation_experiences_multiple_runs",
    "extract_metric_values_over_evaluation_experiences",
    "get_all_tasks_paths",
]
