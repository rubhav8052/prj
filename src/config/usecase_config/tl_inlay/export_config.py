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
from src.config.export_config import ExportConfig

config = ExportConfig()
# config.lora_paths = [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/deepseek/frozen_vit_frozen_llm_90_response_integer/v0-20250904-121302/checkpoint-630"]
config.lora_paths = [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/qwen/frozen_vit_frozen_llm_90_response_integer/v0-20250904-121124/checkpoint-630"]
# Dataset path (JSONL) used for calibration (e.g., training set)
config.dataset_path = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/datasets/drawn_green_box_with_context/tl_inlay_train_v1.jsonl"
# Directory to save the quantized model
# config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/deepseek/exported/v0-20250904-121302/"
config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/qwen/exported/v0-20250904-121124/"