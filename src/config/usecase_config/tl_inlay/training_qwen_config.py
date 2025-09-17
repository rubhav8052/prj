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
from src.config.training_config import TrainingConfig

config = TrainingConfig()
config.model_id = 'Qwen/Qwen2-VL-2B-Instruct'
config.swift_libs_path = '/qwen_libs'
config.train_dataset_paths = [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/datasets/drawn_green_box_with_context/tl_inlay_train_v1.jsonl"]
config.val_dataset_paths = [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/datasets/drawn_green_box_with_context/tl_inlay_val_v1.jsonl"]
config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/traffic_lights_inlay/qwen/frozen_vit_frozen_llm_90_response_integer/"