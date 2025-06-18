import json
import os
from os import PathLike
from typing import Callable, Any


class ConfigParser:

    __standardizer_dict__: dict[str, Callable] = {}
    __parsing_dict__: dict[str, Callable] = {}

    @property
    def standardizer_dict(self):
        return self.__standardizer_dict__
    
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
    def extractable(self): # TODO Do we need to keep it?
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
    def standardizer(cls, key: str, *, replace: bool = False):
        def decorator(func):
            if (key in cls.__standardizer_dict__) and not replace:
                raise KeyError(f"Handler for {key} already registered!")
            cls.__standardizer_dict__[key] = func
            return func
        return decorator

    @classmethod
    def processor(cls, key: str, *, replace: bool = False):
        def decorator(func):
            if (key in cls.__parsing_dict__) and not replace:
                raise KeyError(f"Handler for {key} already registered!")
            cls.__parsing_dict__[key] = func
            return func
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
            self.raw_config = None
        elif isinstance(config_path_or_data, dict):
            self.config_path = None
            self.config = config_path_or_data
            self.raw_config = config_path_or_data.copy()
        else:
            raise TypeError(f"Invalid type for config_path: {type(config_path_or_data)}")
        self.task_id = task_id
        self.__standardized_config = False

    def reset(self):
        self.config_path = None
        self.config = None
        self.raw_config = None
        self.__standardized_config = False

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
        self._standardize_config()
        return True

    def _standardize_config(self):
        if self.config is None:
            raise ValueError("Configuration is not loaded.")
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: '{key}' in configuration.")
            key_standardizer: Callable | None = self.__standardizer_dict__.get(key, None)
            if not key_standardizer:
                raise RuntimeError(f"Config standardizer for {key} not found.")
            default_cfg = key_standardizer(self.raw_config, key) # First the whole config, then the key
            if default_cfg is None:
                raise ValueError(f"Validation failed for key '{key}': {default_cfg}")
            else:
                self.raw_config[key] = default_cfg
        for key in self.optional_keys:
            if key not in self.config:
                self.raw_config[key] = {}
            else:
                key_standardizer: Callable | None = self.__standardizer_dict__.get(key, None)
                if key_standardizer:
                    default_cfg = key_standardizer(self.config, key)
                    if default_cfg is None:
                        raise ValueError(f"Validation failed for key '{key}': {default_cfg}")
                    else:
                        self.raw_config[key] = default_cfg
        self.__standardized_config = True
        self.config = self.raw_config.copy()
        return self.raw_config
    
    def process_config(self):
        """
        Applies handlers to given config data. After it, self.raw_config will contain raw config data with
        filled default values, while self.config will contain (completely) processed data.
        """
        if self.config is None:
            raise ValueError("Configuration is not loaded.")
        elif not self.__standardized_config:
            raise ValueError("Configuration not standardized.")
        all_keys = self.required_keys + self.optional_keys
        for key in all_keys:
            key_handler: Callable | None = self.__parsing_dict__.get(key, lambda x: x) # By default an "identity function"
            if key_handler is None:
                raise RuntimeError(f"Config handler for {key} not found.")
            validation_result = key_handler(self.config[key], task_id=self.task_id, **self.config)
            if validation_result is None:
                raise ValueError(f"Validation failed for key '{key}': {validation_result}")
            else:
                self.config[key] = validation_result
        for key in self.extractable:
            data = self.config.pop(key)
            self.config.update(data)
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

    def get_raw_config(self):
        return {k: v for k, v in self.raw_config.items()}


__all__ = ['ConfigParser']