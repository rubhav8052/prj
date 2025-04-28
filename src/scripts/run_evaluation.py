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

from src.config.evaluation_config import EvaluationConfig
from src.evaluation.metrics import run_evaluation
from src.utils.config_loader import load_config # Assuming a helper function

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation on Model Inference Outputs")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the Python configuration file for evaluation (e.g., configs/my_eval_config.py)",
    )
    args = parser.parse_args()

    # Load the configuration
    config: EvaluationConfig = load_config(args.config, EvaluationConfig)

    print("Configuration loaded:")
    print(config)

    # Run the evaluation pipeline
    run_evaluation(config)

if __name__ == "__main__":
    main() 