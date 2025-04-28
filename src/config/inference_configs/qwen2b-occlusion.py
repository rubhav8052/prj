# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================

import os
from src.config.swift_inference_config import SwiftInferenceConfig

config = SwiftInferenceConfig()
config.model_id = 'Qwen/Qwen2-VL-2B-Instruct'
config.swift_libs_path = '/qwen_libs'
config.lora_path = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/vehicle/occlusion/Qwen2-VL-2B-Instruct/frozen_vit_frozen_llm_90/v0-20250407-094958/checkpoint-582"
config.dataset_path = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/datasets/drawn_green_box_with_context/type_val_v1.jsonl"
config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/inference_outputs/occlusion/qwen2b/"