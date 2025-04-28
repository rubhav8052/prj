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
from typing import List, Optional, Dict

@dataclass
class TrainingConfig:
    """Configuration for Swift SFT training."""
    # Model identifier (Hugging Face or ModelScope)
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct" # Example: "Qwen/Qwen2-VL-7B-Instruct", "deepseek-ai/deepseek-vl2-tiny"
    # Training dataset paths (JSONL format)
    train_dataset_paths: List[str] = field(default_factory=lambda: [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/datasets/drawn_green_box_with_context/occlusion_train_v1.jsonl"])
    # Validation dataset paths (JSONL format)
    val_dataset_paths: List[str] = field(default_factory=lambda: [os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/datasets/drawn_green_box_with_context/occlusion_val_v1.jsonl"])
    # Directory to save checkpoints and logs
    output_dir: str = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+"/attribute_labeling/temp/vehicle/occlusion/deepseek/frozen_vit_frozen_llm_90_response_integer/"
    # Training type (e.g., 'lora', 'full')
    train_type: str = "lora"
    # LoRA rank (if train_type is 'lora')
    lora_rank: int = 256
    # Freeze Vision Transformer weights
    freeze_vit: bool = True
    # Freeze a ratio of LLM parameters (e.g., 0.9 for 90%)
    freeze_parameters_ratio: Optional[float] = 0.9 # Set to None to train all unfrozen LLM params
    # Number of training epochs
    num_train_epochs: int = 3
    # Data type for training (e.g., 'bfloat16', 'float16', 'float32')
    torch_dtype: str = "bfloat16"
    # Attention implementation (e.g., 'flash_attn', 'sdpa', None)
    attn_impl: Optional[str] = None # Example: 'flash_attn' for Qwen
    # Strategy for saving checkpoints ('epoch', 'steps')
    save_strategy: str = "epoch"
    # Number of steps between saves if save_strategy is 'steps'
    save_steps: Optional[int] = None
    # Batch size per device for training
    batch_size: int = 1 # Default from original scripts, adjust as needed
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 16 # Default from swift, adjust as needed
    # Learning rate
    learning_rate: float = 1e-4 # Default from swift, adjust as needed
    # Maximum sequence length
    max_length: int = 2048 # Default from swift, adjust as needed
    # Use deepspeed for distributed training (path to config or True/False)
    deepspeed: Optional[str] = None # Example: 'default_zero2'
    # Additional arguments to pass to Swift TrainArguments
    extra_swift_args: Dict[str, any] = field(default_factory=dict)
    # Environment variables to set
    env_vars: Dict[str, str] = field(default_factory=lambda: {
        "MODELSCOPE_CACHE": os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/modelscope',
        "OMP_NUM_THREADS": "14",
        "CUDA_VISIBLE_DEVICES": "0"
    })

    swift_libs_path: Optional[str] = "/deepseek_libs" if "deepseek" in model_id else "/qwen_libs"


    # --- Example Configurations (Comments) ---
    # Occlusion - DeepSeek - Frozen ViT, 90% LLM Frozen
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny"
    # train_dataset_paths: List[str] = ["occlusion_train_v2.jsonl"]
    # val_dataset_paths: List[str] = ["occlusion_val_v2.jsonl"]
    # output_dir: str = ".../occlusion/deepseek/frozen_vit_frozen_llm_90_response_integer/"
    # freeze_vit: bool = True
    # freeze_parameters_ratio: Optional[float] = 0.9

    # Occlusion - DeepSeek - Full ViT, Full LLM
    # model_id: str = "deepseek-ai/deepseek-vl2-tiny"
    # train_dataset_paths: List[str] = ["occlusion_train_v2.jsonl"]
    # val_dataset_paths: List[str] = ["occlusion_val_v2.jsonl"]
    # output_dir: str = ".../occlusion/deepseek/vit_llm_full_response_integer/"
    # freeze_vit: bool = False
    # freeze_parameters_ratio: Optional[float] = None

    # Vehicle Type - Qwen2-VL-7B - Frozen ViT, 99% LLM Frozen
    # model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # train_dataset_paths: List[str] = ["vehicle_type_train.jsonl"]
    # val_dataset_paths: List[str] = ["vehicle_type_val.jsonl"]
    # output_dir: str = ".../vehicle_type/Qwen2-VL-7B-Instruct/frozen_vit_frozen_llm_99_full_response_integer/"
    # freeze_vit: bool = True
    # freeze_parameters_ratio: Optional[float] = 0.99
    # attn_impl: Optional[str] = 'flash_attn' 