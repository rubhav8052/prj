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
from src.config.data_prep_config import DataPrepConfig
    
config = DataPrepConfig()
config.input_csv_paths = [os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/attribute_labeling/tl_inlays2/full_dataset_2.csv']
# Base path for resolving relative image paths within CSVs
config.image_base_dir = os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/attribute_labeling/tl_inlays2/'
# Directory to save output JSONL files
config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/traffic_lights_inlay/datasets/drawn_green_box_with_context/'
# Column names from CSV
config.attribute_col = "inlay"
# Prompt file for attribute task (used if generating attribute dataset)
config.attribute_prompt_file = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+f'/attribute_labeling/traffic_lights_inlay/datasets/drawn_green_box_with_context/prompts/traffic_lights_inlay_prompt.md' 
config.img_sha_col = "frame_sha"
config.test_size = 0.10
#ignore crops having height/width lower than this
config.min_height_width_pixel = 5

# Output filenames
config.attribute_train_file = "tl_inlay_train_v1.jsonl"
config.attribute_val_file = "tl_inlay_val_v1.jsonl"
config.attribute_test_file = "tl_inlay_test_v1.jsonl"
