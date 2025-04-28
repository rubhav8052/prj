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

import argparse
import os
import importlib
from src.config.inference_config import IMAGE_FOLDER, DEEPSEEK_CONFIG, QWEN_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with selected VLM model")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["deepseek", "qwen", "all"], 
        default="all",
        help="Model to run inference with (deepseek, qwen, or all)"
    )
    parser.add_argument(
        "--image_folder", 
        type=str, 
        default=IMAGE_FOLDER,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--prompt_module",
        type=str,
        default=None,
        help="Module path to the prompt (e.g., src.prompts.custom_prompt)"
    )
    parser.add_argument(
        "--prompt_variable",
        type=str,
        default="prompt",
        help="Variable name in the module containing the prompt"
    )
    return parser.parse_args()

def load_prompt(module_path, variable_name):
    """Dynamically load a prompt from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, variable_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading prompt: {e}")
        return None

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model in ["deepseek", "all"]:
        from src.inference.local_lvms.deepseek_inference import DeepseekInference
        
        print("\n=== Running Deepseek Inference ===")
        config = DEEPSEEK_CONFIG.copy()
        config["output_file"] = os.path.join(args.output_dir, os.path.basename(config["output_file"]))
        
        # Override prompt if specified in arguments
        if args.prompt_module and args.prompt_variable:
            config["prompt_module"] = args.prompt_module
            config["prompt_variable"] = args.prompt_variable
        
        # Load the prompt
        prompt = load_prompt(config["prompt_module"], config["prompt_variable"])
        if prompt:
            config["prompt"] = prompt
            deepseek = DeepseekInference(config)
            deepseek.batch_inference(args.image_folder)
        else:
            print("Failed to load prompt for Deepseek model. Skipping.")
    
    if args.model in ["qwen", "all"]:
        from src.inference.local_lvms.qwen_inference import QwenInference
        
        print("\n=== Running Qwen Inference ===")
        config = QWEN_CONFIG.copy()
        config["output_file"] = os.path.join(args.output_dir, os.path.basename(config["output_file"]))
        
        # Override prompt if specified in arguments
        if args.prompt_module and args.prompt_variable:
            config["prompt_module"] = args.prompt_module
            config["prompt_variable"] = args.prompt_variable
        
        # Load the prompt
        prompt = load_prompt(config["prompt_module"], config["prompt_variable"])
        if prompt:
            config["prompt"] = prompt
            qwen = QwenInference(config)
            qwen.batch_inference(args.image_folder)
        else:
            print("Failed to load prompt for Qwen model. Skipping.")

if __name__ == "__main__":
    main() 