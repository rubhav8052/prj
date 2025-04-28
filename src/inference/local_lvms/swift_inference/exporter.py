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
from typing import Optional
from src.config.export_config import ExportConfig

# Swift Python SDK imports
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

        print(f"Quantization successful. Model saved to: {config.output_dir}")

    except Exception as e:
        print("Quantization failed due to an exception:")
        print(e)
        sys.exit(1)
