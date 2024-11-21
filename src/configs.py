import json
import os


class ConfigParser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}

    def load_config(self):
        """Load and parse the configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")

        with open(self.config_path, 'r') as file:
            try:
                self.config = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

    def validate_config(self):
        """Validate the configuration for mutual exclusivity and required keys."""
        # Check required top-level keys
        required_keys = ['architecture', 'strategy', 'scheduler', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key: '{key}' in configuration.")

        # Validate architecture parameters
        self._validate_architecture()

        # Validate strategy parameters
        self._validate_strategy()

        # Validate scheduler parameters
        self._validate_scheduler()

        # Validate output columns
        self._validate_output()

    def _validate_architecture(self):
        """Validate architecture configuration."""
        architecture = self.config.get('architecture', {})
        if 'model_type' not in architecture:
            raise ValueError("Missing 'model_type' in architecture configuration.")
        if 'parameters' not in architecture or not isinstance(architecture['parameters'], dict):
            raise ValueError("Architecture parameters must be defined as a dictionary.")

    def _validate_strategy(self):
        """Validate strategy configuration with mutual exclusivity."""
        strategy = self.config.get('strategy', {})
        if 'type' not in strategy:
            raise ValueError("Missing 'type' in strategy configuration.")
        if 'parameters' not in strategy or not isinstance(strategy['parameters'], dict):
            raise ValueError("Strategy parameters must be defined as a dictionary.")

        strategy_type = strategy['type']
        supported_strategies = ['EWC', 'Replay', 'LAMAL']
        if strategy_type not in supported_strategies:
            raise ValueError(f"Unsupported strategy type: '{strategy_type}'. Supported types: {supported_strategies}")

        # Ensure mutual exclusivity
        if len(strategy['parameters']) > 1:
            raise ValueError(
                f"Mutually exclusive parameters found in strategy '{strategy_type}'. Only one set of parameters is allowed.")

    def _validate_scheduler(self):
        """Validate scheduler configuration."""
        scheduler = self.config.get('scheduler', {})
        if 'type' not in scheduler:
            raise ValueError("Missing 'type' in scheduler configuration.")
        if scheduler['type'] not in ['StepLR', 'CosineAnnealing', 'Exponential']:
            raise ValueError("Unsupported scheduler type. Supported types: 'StepLR', 'CosineAnnealing', 'Exponential'.")
        if 'parameters' not in scheduler or not isinstance(scheduler['parameters'], dict):
            raise ValueError("Scheduler parameters must be defined as a dictionary.")

    def _validate_output(self):
        """Validate output configuration."""
        output = self.config.get('output', {})
        if 'columns' not in output or not isinstance(output['columns'], list):
            raise ValueError("Output columns must be defined as a list.")
        if not output['columns']:
            raise ValueError("Output columns list cannot be empty.")

    def get_config(self):
        """Return the validated configuration."""
        return self.config


# Example usage
if __name__ == "__main__":
    try:
        # Path to your configuration JSON file
        config_file = "experiment_config.json"

        # Create a ConfigParser instance
        parser = ConfigParser(config_file)

        # Load and validate configuration
        parser.load_config()
        parser.validate_config()

        # Access the validated configuration
        config = parser.get_config()
        print("Configuration successfully loaded and validated.")
        print(json.dumps(config, indent=4))
    except Exception as e:
        print(f"Error: {e}")
