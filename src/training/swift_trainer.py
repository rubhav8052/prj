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

from dataclasses import asdict
import os
import sys
from azureml.core import Run
from transformers import TrainerCallback
import swift.plugin
from swift.llm import sft_main, TrainArguments
from swift.utils import get_logger

from src.config.training_config import TrainingConfig
from src.utils.aml_utils import register_aml_model
logger = get_logger()

class AMLLogger(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for name, val in logs.items():
                if isinstance(val, (int, float)):
                    self.run.log(name, val)

swift.plugin.extra_callbacks.append(AMLLogger())                   

def run_swift_sft(config: TrainingConfig):
    """Runs Swift SFT (Supervised Fine-Tuning) based on the configuration."""
    print("Starting Swift SFT training...")
    print(f"   Model ID: {config.model_id}")
    print(f"   Train Datasets: {config.train_dataset_paths}")
    print(f"   Val Datasets: {config.val_dataset_paths}")
    print(f"   Output Dir: {config.output_dir}")
    print(f"   Train Type: {config.train_type}")
    print(f"   Freeze ViT: {config.freeze_vit}")
    print(f"   Freeze LLM Ratio: {config.freeze_parameters_ratio}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Dtype: {config.torch_dtype}")
    print(f"   Max Length: {config.max_length}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"   Deepspeed: {config.deepspeed}")
    print(f"   Extra Swift Args: {config.extra_swift_args}")


    # Validate dataset paths
    all_datasets = config.train_dataset_paths + config.val_dataset_paths
    if not all_datasets:
         raise ValueError("At least one training or validation dataset path must be provided.")

    for d_path in all_datasets:
        if not os.path.exists(d_path):
             raise FileNotFoundError(f"Dataset file not found: {d_path}")

    os.makedirs(config.output_dir, exist_ok=True)

    # --- Prepare Swift SftArguments ---
    # Map TrainingConfig fields to SftArguments
    # Note: Some names might differ slightly or require specific formatting
    swift_args = {
        "model": config.model_id,
        "train_type": config.train_type,
        "dataset": config.train_dataset_paths, # Swift expects a list for 'dataset'
        "val_dataset": config.val_dataset_paths, # Swift expects a list for 'eval_dataset'
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "max_length": config.max_length,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps if config.save_strategy == 'steps' else None, # Only set save_steps if strategy is 'steps'
        "save_total_limit": 3, # Keep last 3 checkpoints
        "logging_steps": 50, # Log every 50 steps (adjust as needed)
        "eval_steps": 50, # Evaluate every 50 steps (adjust as needed)
        "lora_rank": config.lora_rank if config.train_type == 'lora' else 8, # Default LoRA rank if not specified
        "lora_alpha": config.lora_rank * 2 if config.train_type == 'lora' else 16, # Common practice alpha = 2*rank
        "gradient_checkpointing": True, # Enable gradient checkpointing to save memory
        "torch_dtype": config.torch_dtype,
        "fp16": config.torch_dtype == 'float16',
        "bf16": config.torch_dtype == 'bfloat16',
        "deepspeed": config.deepspeed,
        "report_to": ["tensorboard", "azure_ml"], # Report metrics to TensorBoard and Azure ML
        "dataloader_num_workers": 1, # Adjust based on system capabilities
        "eval_strategy": "steps" if config.val_dataset_paths else "no", # Evaluate if val data provided
        "load_best_model_at_end": True if config.val_dataset_paths else False, # Load best model if evaluating
        "metric_for_best_model": "eval_loss" if config.val_dataset_paths else None, # Use eval loss to find best model
        "greater_is_better": False, # Lower eval loss is better
        "neftune_noise_alpha": 5, # Add noise for potentially better generalization
        "optim": "adamw_torch", # Use AdamW optimizer
        "lr_scheduler_type": "cosine", # Use cosine learning rate scheduler
        "warmup_ratio": 0.03, # Warmup ratio
        "weight_decay": 0.01, # Weight decay
        "max_grad_norm": 1.0, # Gradient clipping
        # Add extra arguments from config
        **config.extra_swift_args
    }

    # --- Handle Vision-Language Model Specific Args ---
    if "vl" in config.model_id.lower() or "vision" in config.model_id.lower():
        swift_args["freeze_vit"] = config.freeze_vit
        if config.freeze_parameters_ratio is not None:
            swift_args["freeze_parameters_ratio"] = config.freeze_parameters_ratio
        # Add other VLM specific args if needed, e.g., vl_resampler_type

    # --- Instantiate SftArguments ---
    try:
        arguments = TrainArguments(**swift_args)
    except TypeError as e:
        logger.error(f"Error creating SftArguments. Check for invalid parameters: {e}")
        logger.error(f"Provided Swift Args: {swift_args}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred creating SftArguments: {e}")
        logger.error(f"Provided Swift Args: {swift_args}")
        sys.exit(1)


    # --- Run Swift SFT Main ---
    try:
        logger.info(f"Launching sft_main with arguments: {arguments}")
        results = sft_main(arguments)
        logger.info(f"Swift SFT finished. Results: {results}")
        print("Swift SFT training completed successfully.")
    except Exception as e:
        logger.error(f"Swift SFT training failed with an exception: {e}", exc_info=True)
        print(f"Error during Swift SFT training: {e}")
        sys.exit(1)

    print("\nAttempting to register the exported model in Azure ML...")
    try:
        # Convert config dataclass to dict for tags, ensuring serializability
        tags_dict = {}
        for key, value in asdict(config).items():
                # Convert lists/dicts to strings, handle None, etc.
                if isinstance(value, (list, dict)):
                    tags_dict[key] = str(value)
                    tags_dict[key] = 'workspaceblobstore'+tags_dict[key].split('workspaceblobstore')[-1]
                elif value is None:
                    tags_dict[key] = "None"
                else:
                    tags_dict[key] = str(value) # Ensure all values are strings

        # Generate a model name (example: qwen-7b-instruct-occlusion-merged)
        model_base_name = config.model.split('/')[-1].lower().replace('_', '-')
        # Try to get a meaningful name part from the output path
        output_path_parts = config.output_dir.strip('/').split('/')
        task_or_detail = output_path_parts[-1] if len(output_path_parts) > 1 else "exported"
        model_name = f"{model_base_name}-{task_or_detail}"

        register_aml_model(
            model_path=config.output_dir,
            tags=tags_dict,
            model_name=model_name
        )
    except ImportError:
            print("Warning: Azure ML SDK not found or not configured. Skipping model registration.")
    except Exception as reg_e:
        print(f"Error during Azure ML model registration: {reg_e}")
        # Decide if this error should cause the script to exit
        # sys.exit(1)