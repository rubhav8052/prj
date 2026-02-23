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

# Common configurations
BASE_PATH = os.environ["AZUREML_DATAREFERENCE_azureml_container"]
# AZUREML_DATAREFERENCE_azureml_container
BASE_WORKSPACE_PATH = os.environ['AZUREML_DATAREFERENCE_azureml_container']
IMAGE_FOLDER = BASE_PATH + '/raw_output/tl_inlay/images'
# raw_output/tl_inlay/images/
# IMAGE_FOLDER = BASE_PATH + '/attribute_labeling/tl_inlays2/test_images/'
# /tmp/azureml_data_mount/raw_output/deepseek/exported/v4-20260201-202214/checkpoint-159/"
# raw_output/deepseek/exported/v4-20260201-202214/
# Model specific configurations

# //////////////////////////////////////////////////////////////////////
print(f"BASE_PATH: {BASE_PATH}")
print(f"BASE_WORKSPACE_PATH: {BASE_WORKSPACE_PATH}")
print("-----")
print(f"\nContents of BASE_PATH:")
print(os.listdir(BASE_PATH))

# Check if tmp directory exists
tmp_path = os.path.join(BASE_PATH, "tmp")
if os.path.exists(tmp_path):
    print(f"\nContents of {tmp_path}:")
    print(os.listdir(tmp_path))

# Look for deepseek anywhere
print("\nSearching for 'deepseek' directories:")
for root, dirs, files in os.walk(BASE_PATH):
    if 'deepseek' in root.lower():
        print(f"Found: {root}")
        break
    # Limit depth to avoid long searches
    if root.count(os.sep) - BASE_PATH.count(os.sep) > 5:
        break
model_path = BASE_WORKSPACE_PATH + "/raw_output/deepseek/exported/v4-20260201-202214/"
print(f"Full model path: {model_path}")
print(f"Model path exists: {os.path.exists(model_path)}")
# //////////////////////////////////////////////////////////////////////
DEEPSEEK_CONFIG = {
    "model_name": BASE_WORKSPACE_PATH+"/raw_output/deepseek/exported/v4-20260201-202214/checkpoint-159/",
    "max_model_len": 4096,
    "max_num_seqs": 32,
    "batch_size": 5,
    "output_file": BASE_WORKSPACE_PATH + "/raw_output/deepseek/inference_outputs/v4-20260201-202214/frozen_vit_frozen_llm_90_output.json",
    "prompt_module": "src.prompts.tl_inlay_prompt",
    "prompt_variable": "tl_inlay_prompt"
}

QWEN_CONFIG = {
    "model_name": BASE_WORKSPACE_PATH+"/attribute_labeling/traffic_lights_inlay/qwen/exported/v0-20250904-121124/",
    "max_model_len": 4096,
    "max_num_seqs": 5,
    "batch_size": 16,
    "output_file": BASE_WORKSPACE_PATH + "/attribute_labeling/traffic_lights_inlay/qwen/inference_outputs/v0-20250904-121124/frozen_vit_frozen_llm_90_output.json",
    "prompt_module": "src.prompts.tl_inlay_prompt",
    "prompt_variable": "tl_inlay_prompt"
}

# Common sampling parameters
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "max_tokens": 5000
} 

# import os

# # Common configurations
# BASE_PATH = os.environ["AZUREML_DATAREFERENCE_VDEEPINGESTPROD"]
# BASE_WORKSPACE_PATH = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']
# IMAGE_FOLDER = BASE_PATH + '/attribute_labeling/tl_inlays2/test_images/'

# # Model specific configurations
# DEEPSEEK_CONFIG = {
#     "model_name": BASE_WORKSPACE_PATH+"/attribute_labeling/traffic_lights_inlay/deepseek/exported/v9-20260105-064828/",
#     "max_model_len": 4096,
#     "max_num_seqs": 32,
#     "batch_size": 5,
#     "output_file": BASE_WORKSPACE_PATH + "/attribute_labeling/traffic_lights_inlay/deepseek/inference_outputs/v9-20260105-064828/frozen_vit_frozen_llm_90_output.json",
#     "prompt_module": "src.prompts.tl_inlay_prompt",
#     "prompt_variable": "tl_inlay_prompt"
# }

# QWEN_CONFIG = {
#     "model_name": BASE_WORKSPACE_PATH+"/attribute_labeling/traffic_lights_inlay/qwen/exported/v0-20250904-121124/",
#     "max_model_len": 4096,
#     "max_num_seqs": 5,
#     "batch_size": 16,
#     "output_file": BASE_WORKSPACE_PATH + "/attribute_labeling/traffic_lights_inlay/qwen/inference_outputs/v0-20250904-121124/frozen_vit_frozen_llm_90_output.json",
#     "prompt_module": "src.prompts.tl_inlay_prompt",
#     "prompt_variable": "tl_inlay_prompt"
# }

# # Common sampling parameters
# SAMPLING_PARAMS = {
#     "temperature": 0.7,
#     "top_p": 0.8,
#     "repetition_penalty": 1.05,
#     "max_tokens": 5000
# } 