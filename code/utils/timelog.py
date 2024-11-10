from datetime import datetime


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


__all__ = ["time_logger"]
