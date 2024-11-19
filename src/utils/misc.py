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


__all__ = ["debug_print", "time_logger", "float16", "float32", "float64", "get_dtype_from_str"]
