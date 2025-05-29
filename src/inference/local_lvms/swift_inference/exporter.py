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
import sys
import shutil
from dataclasses import asdict
from typing import Optional
from src.config.export_config import ExportConfig
from src.utils.aml_utils import register_aml_model

# Swift Python SDK imports
from azureml.core import Model
from swift.llm import export_main, ExportArguments


def run_swift_export(config: ExportConfig):
    """Runs Swift model export using the Swift Python SDK."""
    print("Starting Swift model quantization (via Python SDK)...")
    print(f"   Model ID: {config.model}")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Output Dir: {config.output_dir}")
    # print(f"   Bits: {config.quant_bits}")
    # print(f"   Method: {config.quant_method}")
    # print(f"   Batch Size: {config.quant_batch_size}")
        
    # --- Check and delete existing output directory ---
    if os.path.exists(config.output_dir):
        print(f"Warning: Output directory '{config.output_dir}' already exists. Deleting it...")
        try:
            shutil.rmtree(config.output_dir)
            print(f"Successfully deleted existing directory: {config.output_dir}")
        except OSError as e:
            print(f"Error deleting directory {config.output_dir}: {e}")
            print("Please check permissions or manually delete the directory and retry.")
            sys.exit(1) # Exit if deletion fails

    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Calibration dataset not found: {config.dataset_path}")

    try:
        # Build the export arguments
        args = ExportArguments(
            model=config.model,
            # quant_bits=config.quant_bits,
            # quant_method=config.quant_method,
            dataset=config.dataset_path,
            # quant_n_samples=config.quant_n_samples,
            # quant_batch_size=config.quant_batch_size,
            output_dir=config.output_dir,
            merge_lora=config.merge_lora,
            adapters=config.lora_paths if config.merge_lora else [],
         
        )

        # Run the export
        export_main(args)

        if config.merge_lora:
            print(f"LoRA merge successful. Merged model saved to: {config.output_dir}")
        else:
             print(f"Export (without merge) successful. Model saved to: {config.output_dir}")
        export_successful = True

    except Exception as e:
        print("Model export failed due to an exception:")
        print(e)
        # Optionally re-raise or handle differently
        # sys.exit(1) # Consider if failure should stop the whole script

    # --- Register Model in Azure ML if export was successful ---
    if export_successful and config.output_dir.startswith(os.environ.get('AZUREML_DATAREFERENCE_workspaceblobstore', 'dummy_prefix')):
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
            task_or_detail = output_path_parts[-3] if len(output_path_parts) > 1 else "exported"
            model_name = f"{model_base_name}-{task_or_detail}-merged" if config.merge_lora else f"{model_base_name}-{task_or_detail}-exported"

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
    elif not export_successful:
         print("\nSkipping Azure ML model registration because export failed.")
    else:
         print(f"\nSkipping Azure ML model registration: Output directory '{config.output_dir}' does not seem to be an AML path.")

