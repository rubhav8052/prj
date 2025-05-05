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
class DataPrepConfig:
    """Configuration for dataset preparation."""
    # Input data paths (accepts multiple CSVs)
    input_csv_paths: List[str] = field(default_factory=lambda: [
                                  os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data/Vehicles_sampled_2.csv', 
                                  os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data/Vehicles_sampled_3.csv'
                                  ])
    # Base path for resolving relative image paths within CSVs
    image_base_dir: str = os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/object_retrieval/AL_data/vehicle_sampled/'
    # Directory to save output JSONL files
    output_dir: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/vehicle/datasets/drawn_green_box_with_context/'
    # Column names from CSV
    attribute_col: str = "occlusion"
    # Prompt file for attribute task (used if generating attribute dataset)
    attribute_prompt_file: Optional[str] = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+f'/attribute_labeling/vehicle/datasets/drawn_green_box_with_context/prompts/occlusion prompt.md' 
    # Percentage of context to add around bounding box
    context_percent: float = 0.5
    # Test set size for splitting
    test_size: float = 0.20
    # Random state for splitting
    random_state: int = 42
    # Number of threads for image processing
    num_threads: int = 32
    img_sha_col: str = "img_sha"
    bbox_cols: List[str] = field(default_factory=lambda: ["x0", "y0", "x1", "y1"])
    index_col: str = "index" # Assuming 'index' column exists or will be created

    # Output filenames
    attribute_train_file: str = f"{attribute_col}_train_v1.jsonl"
    attribute_val_file: str = f"{attribute_col}_val_v1.jsonl"