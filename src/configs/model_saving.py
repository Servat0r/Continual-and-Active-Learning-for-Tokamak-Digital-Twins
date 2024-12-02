import os
from typing import Any
from datetime import datetime

from .parser import *

MODELS_DIR = os.getenv('MODELS_DIR', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


@ConfigParser.register_handler('start_model_saving')
def start_model_saving_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    save_model = data.get('save_model', False)
    save_model_folder = f"{MODELS_DIR}/{data.get('saved_model_folder', "")}"
    saved_model_name = data.get('saved_model_name', None)
    add_timestamp = data.get('add_timestamp', False)
    if save_model:
        if saved_model_name is None:
            raise ValueError(f"\"saved_model_name\" field not present in configuration")
        if add_timestamp:
            saved_model_name = f"{saved_model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
        saved_model_name = f"{saved_model_name} task_{task_id}"
        return {'saved_model_folder': save_model_folder, 'saved_model_name': saved_model_name}
    else:
        return {}


__all__ = ['start_model_saving_handler', 'MODELS_DIR']
