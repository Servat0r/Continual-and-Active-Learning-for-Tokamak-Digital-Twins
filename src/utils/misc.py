import os
from torch import float16, float32, float64
from datetime import datetime
from dotenv import load_dotenv
from rich import print as dbprint


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


__all__ = [
    "debug_print", "time_logger", "float16", "float32", "float64",
    "get_dtype_from_str", "extract_metric_info", "extract_metric_type",
]
