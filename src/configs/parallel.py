from typing import Any

from .parser import *


@ConfigParser.register_handler('parallel')
def parallel_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    return data # For now, nothing particular


__all__ = ['parallel_handler']
