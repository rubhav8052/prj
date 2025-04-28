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
from typing import Optional, List, Dict

@dataclass
class SwiftRequestConfig:
    """Configuration for Swift inference requests."""
    max_tokens: int = 128 # Max new tokens to generate
    temperature: float = 0.0 # Sampling temperature
    top_p: Optional[float] = None # Nucleus sampling p
    top_k: Optional[int] = None # Top-k sampling k
    repetition_penalty: float = 1.0 # Repetition penalty
    # Add other relevant parameters from swift.llm.RequestConfig if needed

@dataclass
class SwiftInferenceConfig:
    """Configuration for Swift inference."""
    # Base model identifier (Hugging Face or ModelScope)
    model_id: str = "deepseek-ai/deepseek-vl2-tiny" # Example: "Qwen/Qwen2-VL-7B-Instruct"
    # Path to LoRA weights (if using a fine-tuned model), In case of directory it will pick the latest checkpint from the directory
    lora_path: Optional[str] = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/vehicle/occlusion/deepseek/frozen_vit_frozen_llm_90/v11-20250408-181301/checkpoint-9324" 
    # Path to the dataset for inference (JSONL format)
    dataset_path: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/vehicle/datasets/drawn_green_box_with_context/occlusion_val.jsonl'
    # Directory to save inference results
    output_dir: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/inference_outputs/occlusion/deepseek/"
    # Output filename prefix (e.g., 'model_run_name')
    output_filename_prefix: str = "frozen_vit_frozen_llm_90" # Will be saved as f"{output_filename_prefix}_output.jsonl"
    # Batch size for inference
    batch_size: int = 2
    # Data type for inference ('float16', 'bfloat16')
    dtype: str = 'bfloat16' # 'float16' was used in one script
    # Specific template type if needed (otherwise inferred from model)
    template_type: Optional[str] = None
    # Request configuration
    request_config: SwiftRequestConfig = field(default_factory=SwiftRequestConfig)
    # Environment variables to set
    env_vars: Dict[str, str] = field(default_factory=lambda: {
        "MODELSCOPE_CACHE": os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/modelscope", # Example: os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/modelscope'
        "OMP_NUM_THREADS": "14", # Adjust as needed for inference
        "CUDA_VISIBLE_DEVICES": "0"
    })
    # Path to swift libraries if needed (e.g., sys.path.insert)
    swift_libs_path: Optional[str] = "/deepseek_libs" if "deepseek" in model_id else "/qwen_libs"

    # --- Example Configurations (Comments) ---

    # Example 1: Pretrained DeepSeek Occlusion Inference
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny"
    # lora_path: Optional[str] = None
    # dataset_path: str = ".../occlusion_val.jsonl" # Use the correct validation set path
    # output_dir: str = "inference_outputs/occlusion/deepseek/pretrained/"
    # output_filename_prefix: str = "deepseek-vl2-tiny_pretrained"
    # batch_size: int = 2
    # request_config: SwiftRequestConfig = SwiftRequestConfig(max_tokens=1280, temperature=0.0)
    # swift_libs_path: Optional[str] = "/deepseek_libs" # If needed

    # Example 2: Fine-tuned DeepSeek Occlusion Inference (Frozen ViT, 90% LLM Frozen)
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny" # Base model used for fine-tuning
    # lora_path: Optional[str] = ".../occlusion/deepseek/frozen_vit_frozen_llm_90_response_integer/checkpoint-YYYY" # Path to specific checkpoint
    # dataset_path: str = "occlusion_val_v2.jsonl" # Validation set used during fine-tuning
    # output_dir: str = "inference_outputs/occlusion/deepseek/frozen_vit_frozen_llm_90_response_integer/"
    # output_filename_prefix: str = "deepseek_frozen_vit_frozen_llm_90"
    # batch_size: int = 2
    # dtype: str = 'float16' # As used in the original script
    # request_config: SwiftRequestConfig = SwiftRequestConfig(max_tokens=1280, temperature=0.0)
    # swift_libs_path: Optional[str] = "/deepseek_libs" # If needed

    # Example 3: Pretrained Qwen2-VL-7B Occlusion Inference
    # model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # lora_path: Optional[str] = None
    # dataset_path: str = ".../occlusion_val.jsonl" # Use the correct validation set path
    # output_dir: str = "inference_outputs/occlusion/Qwen2-VL-7B-Instruct/pretrained/"
    # output_filename_prefix: str = "Qwen2-VL-7B-Instruct_pretrained"
    # batch_size: int = 2
    # request_config: SwiftRequestConfig = SwiftRequestConfig(max_tokens=1280, temperature=0.0)
    # swift_libs_path: Optional[str] = "/qwen_libs" # If needed

    # Example 4: Fine-tuned Qwen2-VL-7B Occlusion Inference (Frozen ViT, 90% LLM Frozen)
    # model_id: str = "Qwen/Qwen2-VL-7B-Instruct" # Base model used for fine-tuning
    # lora_path: Optional[str] = ".../occlusion/Qwen2-VL-7B-Instruct/frozen_vit_frozen_llm_90_response_integer/checkpoint-ZZZZ" # Path to specific checkpoint
    # dataset_path: str = "occlusion_val_v2.jsonl" # Validation set used during fine-tuning
    # output_dir: str = "inference_outputs/occlusion/Qwen2-VL-7B-Instruct/frozen_vit_frozen_llm_90_response_integer/"
    # output_filename_prefix: str = "Qwen2-VL-7B_frozen_vit_frozen_llm_90"
    # batch_size: int = 2
    # request_config: SwiftRequestConfig = SwiftRequestConfig(max_tokens=128, temperature=0.0) # Note: max_tokens was 128 in original qwen_inference-Copy1.py
    # swift_libs_path: Optional[str] = "/qwen_libs" # If needed 