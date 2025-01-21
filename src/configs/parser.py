import json
import os
from os import PathLike
from typing import Callable, Any


class ConfigParser:

    __parsing_dict__: dict[str, Callable] = {}

    @property
    def parsing_dict(self):
        return self.__parsing_dict__

    @property
    def required_keys(self):
        return [
            'general',          # Generic info
            'dataset',          # Dataset info
            'architecture',     # Model architecture info
            'strategy',         # CL Strategy info
            'loss',             # Loss info
            'optimizer'         # Optimizer info
        ]

    @property
    def extractable(self):
        return ['general', 'dataset']

    @property
    def optional_keys(self):
        return [
            'scheduler',            # Scheduler info
            'early_stopping',       # Early Stopping info
            'validation_stream',    # Validation Stream usage info
            'plugins',              # (Other) Plugins info
            'transform',            # Input Transforms
            'target_transform',     # Target Transforms
            'start_model_saving',   # Start Model Saving
            'active_learning',      # Active Learning
            'parallel',             # Parallel Execution
        ]

    @classmethod
    def register_handler(cls, key: str):
        def decorator(func):
            cls.__parsing_dict__[key] = func
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def __init__(
            self, config_path_or_data: str | PathLike | dict[str, Any] = None,
            task_id: int = 0
    ):
        if any([
            isinstance(config_path_or_data, str), isinstance(config_path_or_data, PathLike)
        ]):
            self.config_path = config_path_or_data
            self.config = None
        elif isinstance(config_path_or_data, dict):
            self.config_path = None
            self.config = config_path_or_data
        else:
            raise TypeError(f"Invalid type for config_path: {type(config_path_or_data)}")
        self.task_id = task_id

    def reset(self):
        self.config_path = None
        self.config = None

    def load_config(self):
        if self.config is not None:
            return
        if self.config_path is None:
            raise ValueError("Configuration path is not set.")
        elif not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")
        with open(self.config_path, 'r') as file:
            try:
                self.config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        return True

    def process_config(self):
        if self.config is None:
            raise ValueError("Configuration is not loaded.")
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: '{key}' in configuration.")
            key_handler: Callable | None = self.__parsing_dict__.get(key, None)
            if not key_handler:
                raise RuntimeError(f"Config handler for {key} not found.")
            validation_result = key_handler(self.config[key], task_id=self.task_id, **self.config)
            if validation_result is None:
                raise ValueError(f"Validation failed for key '{key}': {validation_result}")
            else:
                self.config[key] = validation_result
        for key in self.extractable:
            data = self.config.pop(key)
            self.config.update(data)
        for key in self.optional_keys:
            print(f"Parsing key = {key}")
            if key not in self.config:
                self.config[key] = {}
            else:
                key_handler: Callable | None = self.__parsing_dict__.get(key, None)
                print(f"key_handler for key = {key} is: {key_handler}")
                if key_handler:
                    validation_result = key_handler(self.config[key], task_id=self.task_id, **self.config)
                    if validation_result is None:
                        raise ValueError(f"Validation failed for key '{key}': {validation_result}")
                    else:
                        self.config[key] = validation_result
        return self.config

    def __getitem__(self, item):
        if self.config is None:
            raise ValueError("Configuration is not loaded.")
        else:
            return self.config[item]

    def __setitem__(self, key, value):
        if self.config is None:
            raise ValueError("Configuration is not loaded.")
        else:
            self.config[key] = value

    def get_config(self):
        return {k: v for k, v in self.config.items()}


__all__ = ['ConfigParser']