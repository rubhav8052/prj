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

import argparse
import os
import sys

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

from src.config.export_config import ExportConfig
from src.utils.file_utils import set_environment_variables
from src.utils.config_loader import load_config # Assuming a helper function

def main():
    parser = argparse.ArgumentParser(description="Run Swift Model Quantization")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Python configuration file for quantization (e.g., configs/my_quant_config.py)",
    )
    args = parser.parse_args()

    # Load the configuration
    config: ExportConfig = load_config(args.config, ExportConfig)

    print("Configuration loaded:")
    print(config)

    if not os.path.exists('images'):
        os.symlink(os.path.dirname(config.dataset_path)+'/images', 'images')

    if config.swift_libs_path and config.swift_libs_path not in sys.path:
        print(f"Adding Swift libs path to sys.path: {config.swift_libs_path}")
        sys.path.insert(0, config.swift_libs_path)

    # Set environment variables from config
    set_environment_variables(config.env_vars)

    from src.inference.local_lvms.swift_inference.exporter import run_swift_export
    
    # Run the quantization
    run_swift_export(config)

if __name__ == "__main__":
    main() 