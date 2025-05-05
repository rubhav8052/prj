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
from typing import Optional, Dict, List

@dataclass
class ExportConfig:
    """Configuration for Swift model quantization."""
    # Model identifier (Hugging Face or ModelScope)
    model: str = "deepseek-ai/deepseek-vl2-tiny" # Example: "Qwen/Qwen2-VL-7B-Instruct"

    lora_paths: Optional[List[str]] = field(default_factory=lambda:[os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/vehicle/occlusion/deepseek/frozen_vit_frozen_llm_90/v11-20250408-181301/checkpoint-9324"])
    
    # Dataset path (JSONL) used for calibration (e.g., training set)
    dataset_path: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/datasets/drawn_green_box_with_context/occlusion_train_v1.jsonl"
    # Directory to save the quantized model
    output_dir: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/vehicle/occlusion/exported/deepseek/"
    # Quantization bits (e.g., 4, 8)
    # quant_bits: int = 4
    # Quantization method (e.g., 'gptq', 'awq')
    # quant_method: str = "gptq"
    # Batch size for quantization calibration
    # quant_batch_size: int = 1
    # Number of samples for calibration
    # quant_n_samples: int = 256 # Default, adjust as needed
    # Environment variables to set
    env_vars: Dict[str, str] = field(default_factory=lambda: {
        "MODELSCOPE_CACHE": os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/modelscope", # Example: os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/modelscope'
        "OMP_NUM_THREADS": "14",
        "CUDA_VISIBLE_DEVICES": "0"
    })

    merge_lora: bool = True
    
    swift_libs_path: Optional[str] = "/deepseek_libs" if "deepseek" in model else "/qwen_libs"

    # --- Example Configurations (Comments) ---
    # DeepSeek Occlusion - GPTQ 4bit
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny"
    # dataset_path: str = "occlusion_train.jsonl"
    # output_dir: str = ".../occlusion/deepseek/quantized/gptq_4bit/"
    # quant_bits: int = 4
    # quant_method: str = "gptq"

    # DeepSeek Occlusion - GPTQ 8bit
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny"
    # dataset_path: str = "occlusion_train.jsonl"
    # output_dir: str = ".../occlusion/deepseek/quantized/gptq_8bit/"
    # quant_bits: int = 8
    # quant_method: str = "gptq"

    # Qwen Vehicle Type - GPTQ 4bit (Example, adjust model_id/paths)
    # model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # dataset_path: str = "vehicle_type_train.jsonl"
    # output_dir: str = ".../vehicle_type/Qwen2-VL-7B-Instruct/quantized/gptq_4bit/"
    # quant_bits: int = 4
    # quant_method: str = "gptq" 