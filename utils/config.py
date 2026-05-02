"""
============================================================
IntelliTraffic – Configuration Loader
============================================================
Loads config.yaml and provides a singleton Config object
used by all modules across the pipeline.
"""

import os
import yaml


class Config:
    """
    Singleton configuration manager.
    Loads settings from a YAML file and provides dot-notation
    access via nested dictionaries.
    """

    _instance = None
    _config = None

    def __new__(cls, config_path: str = None):
        """Ensure only one Config instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml. If None, looks for config.yaml
                         in the project root directory.
        """
        if Config._config is not None and config_path is None:
            return  # Already loaded

        if config_path is None:
            # Default: look for config.yaml in the project root
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.yaml"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            Config._config = yaml.safe_load(f)

        print(f"[Config] Loaded configuration from: {config_path}")

    def get(self, *keys, default=None):
        """
        Retrieve a nested config value using dot-separated keys.

        Example:
            config.get("detection", "confidence")  → 0.4
            config.get("speed", "speed_limit", default=60)  → 60

        Args:
            *keys: Sequence of nested keys to traverse.
            default: Default value if key path doesn't exist.

        Returns:
            The config value, or default if not found.
        """
        value = Config._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def raw(self) -> dict:
        """Return the raw config dictionary."""
        return Config._config

    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
        cls._config = None

    def __repr__(self):
        return f"Config({list(Config._config.keys()) if Config._config else 'not loaded'})"
