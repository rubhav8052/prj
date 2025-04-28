# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================

import importlib.util
import os
from typing import Type, TypeVar, Optional

T = TypeVar('T')

def load_config(config_path: Optional[str], config_class: Type[T]) -> T:
    """
    Loads a configuration class instance from a Python file, or returns
    a default instance if config_path is None.

    Args:
        config_path: Path to the Python configuration file, or None to get defaults.
        config_class: The expected dataclass or class type of the config.

    Returns:
        An instance of config_class populated from the file or with defaults.

    Raises:
        FileNotFoundError: If config_path is provided but the file doesn't exist.
        AttributeError: If the file is loaded but the expected 'config' object is not found.
        TypeError: If the loaded object is not an instance of config_class.
        ImportError: If the configuration file cannot be loaded or executed.
    """
    # If no config path is provided, return a default instance of the class
    if config_path is None or config_path == 'None':
        print(f"No config path provided. Returning default instance of {config_class.__name__}.")
        try:
            # This assumes the config_class can be instantiated without arguments
            # (i.e., all fields have defaults or are Optional)
            return config_class()
        except TypeError as e:
            raise TypeError(
                f"Could not create a default instance of {config_class.__name__}. "
                f"Ensure all fields have default values or handle instantiation differently. Error: {e}"
            )

    # Proceed with loading from file if config_path is provided
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Create a module spec from the file path
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {config_path}")

    # Create a new module based on the spec
    config_module = importlib.util.module_from_spec(spec)

    # Execute the module code
    try:
        spec.loader.exec_module(config_module)
    except Exception as e:
        raise ImportError(f"Failed to execute configuration file {config_path}: {e}")

    # Look for a variable named 'config' holding the configuration instance
    if not hasattr(config_module, 'config'):
        raise AttributeError(f"Configuration file {config_path} must define a variable named 'config'")

    config_instance = getattr(config_module, 'config')

    # Check if the loaded config is an instance of the expected class
    if not isinstance(config_instance, config_class):
        raise TypeError(f"Configuration object in {config_path} is not an instance of {config_class.__name__}. Found type: {type(config_instance).__name__}")

    print(f"Loaded configuration from: {config_path}")
    return config_instance 