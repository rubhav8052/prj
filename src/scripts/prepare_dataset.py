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
import importlib
import os
import sys

# Add src directory to Python path to allow importing config/modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir) # Assuming scripts is one level below src
sys.path.insert(0, src_dir)

from src.config.data_prep_config import DataPrepConfig
from data_preparation.prepare_usecase_dataset import prepare_datasets
from src.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Prepare Vehicle Datasets (Images and JSONL)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Python configuration file (e.g., configs/my_data_prep_config.py)",
    )
    args = parser.parse_args()

    # Load the configuration dynamically
    config: DataPrepConfig = load_config(args.config, DataPrepConfig)

    print("Configuration loaded:")
    print(config)

    # Run the dataset preparation pipeline
    prepare_datasets(config)

if __name__ == "__main__":
    main() 