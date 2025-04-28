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

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Folder containing inference output JSONL files (e.g., *_output.jsonl)
    inference_output_folder: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/vehicle/inference_outputs/vehicle_type/deepseek/'
    # Directory to save evaluation results (plots, CSVs)
    evaluation_output_dir: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/temp/vehicle/evaluation_results/vehicle_type/deepseek/'
    # Task type ('occlusion', 'classification', or 'auto' to infer)
    task_type: str = "auto" # Options: 'occlusion', 'classification', 'auto'
    # Bin size for occlusion confusion matrix
    occlusion_bin_size: int = 10
    # Output summary CSV filename
    summary_csv_filename: str = "model_evaluation_summary.csv"
    # Class labels for classification confusion matrix (optional, inferred if None)
    classification_labels: Optional[List[str]] = None
    # Mapping for class normalization (e.g., {'trucktrailer': 'truck'})
    class_normalization_map: dict = field(default_factory=lambda: {'trucktrailer': 'truck'})

    # --- Example Configurations (Comments) ---
    # Occlusion Evaluation
    # inference_output_folder: str = "inference_outputs/occlusion/deepseek/"
    # evaluation_output_dir: str = "evaluation_results/occlusion/deepseek/"
    # task_type: str = "occlusion"
    # summary_csv_filename: str = "occlusion_evaluation_summary.csv"

    # Vehicle Type Evaluation
    # inference_output_folder: str = "inference_outputs/vehicle_type/deepseek/"
    # evaluation_output_dir: str = "evaluation_results/vehicle_type/deepseek/"
    # task_type: str = "classification"
    # summary_csv_filename: str = "vehicle_type_evaluation_summary.csv"
    # class_normalization_map: dict = field(default_factory=lambda: {'trucktrailer': 'truck'}) 