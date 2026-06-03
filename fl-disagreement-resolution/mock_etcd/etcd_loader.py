"""Configuration loader for federated learning."""

import os
import json
from datetime import datetime

from mock_etcd.config_schema import validate_config


class MockEtcdLoader:
    """Loads and manages configuration for federated learning."""

    def __init__(self, config_path="mock_etcd/configuration.json"):
        """Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate()
        self.results_dir = self._setup_results_dir()

    def _load_config(self):
        """Load configuration from JSON file.

        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")

    def _validate(self):
        """Validate the loaded configuration and raise on errors."""
        errors = validate_config(self.config)
        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

    def _setup_results_dir(self):
        """Setup and return the results directory path.

        Returns:
            str: Path to the results directory
        """
        results_config = self.config.get("results", {})
        experiment_type = self.config.get("experiment", {}).get("type", "unknown")
        directory_suffix = results_config.get("directory_suffix", "")

        #If a custom directory is specified, use it
        if results_config.get("custom_dir"):
            results_dir = results_config["custom_dir"]
        #Otherwise, use a timestamped directory in the base directory if specified
        elif results_config.get("use_timestamped_dir", True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = os.path.join(
                results_config.get("base_dir", "results"),
                f"fl_simulation_{timestamp}_{experiment_type}{directory_suffix}"
            )
        # Otherwise, just use the base directory
        else:
            results_dir = results_config.get("base_dir", "results")

        # Create the directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Create the output directory
        output_dir = os.path.join(
            results_dir,
            results_config.get("structure", {}).get("output_dir", "output")
        )
        os.makedirs(output_dir, exist_ok=True)

        #Create output directories for server
        server_output_dir = os.path.join(output_dir, "server")
        os.makedirs(server_output_dir, exist_ok=True)
        os.makedirs(os.path.join(server_output_dir, "plots"), exist_ok=True)

        return results_dir

    def get_experiment_config(self):
        """Get experiment configuration.

        Returns:
            dict: Experiment configuration
        """
        return self.config.get("experiment", {})

    def get_data_config(self):
        """Get data configuration.

        Returns:
            dict: Data configuration
        """
        return self.config.get("data", {})

    def get_training_config(self):
        """Get training configuration.

        Returns:
            dict: Training configuration
        """
        return self.config.get("training", {})

    def get_results_config(self):
        """Get results configuration.

        Returns:
            dict: Results configuration
        """
        config = self.config.get("results", {})
        #Add the computed results directory
        config["results_dir"] = self.results_dir
        return config

    def get_path(self, *path_components):
        """Get a path within the results directory.

        Args:
            *path_components: Components of the path to join

        Returns:
            str: Full path
        """
        return os.path.join(self.results_dir, *path_components)

    def get_train_dir(self, experiment_type=None):
        """Get the training data directory for the specified experiment type.

        Args:
            experiment_type: Type of experiment (if None, use the one from config)

        Returns:
            str: Path to the training data directory
        """
        if experiment_type is None:
            experiment_type = self.config.get("experiment", {}).get("type")

        train_dirs = self.config.get("data", {}).get("train_dir", {})
        return train_dirs.get(experiment_type)

    def get_test_dir(self, experiment_type=None):
        """Get the test data directory for the specified experiment type.

        Args:
            experiment_type: Type of experiment (if None, use the one from config)

        Returns:
            str: Path to the test data directory
        """
        if experiment_type is None:
            experiment_type = self.config.get("experiment", {}).get("type")

        test_dirs = self.config.get("data", {}).get("test_dir", {})
        return test_dirs.get(experiment_type)

    def get_unlearning_config(self):
        """Get unlearning configuration.

        Returns:
            dict: Unlearning configuration
        """
        return self.config.get("unlearning", {})
