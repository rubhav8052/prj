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

from src.config.swift_inference_config import SwiftInferenceConfig
from src.utils.file_utils import set_environment_variables
from src.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Run Inference using Swift Engine")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Python configuration file for Swift inference (e.g., configs/my_swift_inference_config.py)",
    )
    args = parser.parse_args()

    # --- Load the configuration early ---
    config: SwiftInferenceConfig = load_config(args.config, SwiftInferenceConfig)

    print("Configuration loaded:")
    print(config)
    if not os.path.exists('images'):
        os.symlink(os.path.dirname(config.dataset_path)+'/images', 'images')
    # --- Modify sys.path BEFORE importing the engine ---
    if config.swift_libs_path and config.swift_libs_path not in sys.path:
        print(f"Adding Swift libs path to sys.path: {config.swift_libs_path}")
        sys.path.insert(0, config.swift_libs_path)
        
    # This import happens *after* sys.path might have been modified
    from src.inference.local_lvms.swift_inference.engine import run_swift_inference

    # Set environment variables from config
    set_environment_variables(config.env_vars)

    # Run the inference
    run_swift_inference(config)

if __name__ == "__main__":
    main() 