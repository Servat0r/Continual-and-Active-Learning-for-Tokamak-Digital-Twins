from torch import float16, float32, float64
from datetime import datetime


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


def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        print(f"[{func.__name__}] Start Time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        result = func(*args, **kwargs)
        end_time = datetime.now()
        print(f"[{func.__name__}] End Time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        elapsed_time = end_time - start_time
        print(f"[{func.__name__}] Elapsed Time:", elapsed_time)
        return result
    return wrapper


__all__ = ["time_logger", "float16", "float32", "float64", "get_dtype_from_str"]
