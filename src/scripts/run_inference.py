import argparse
import os
from src.config.inference_config import IMAGE_FOLDER, DEEPSEEK_CONFIG, QWEN_CONFIG
from src.inference.local_lvms.deepseek_inference import DeepseekInference
from src.inference.local_lvms.qwen_inference import QwenInference

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model in ["deepseek", "all"]:
        print("\n=== Running Deepseek Inference ===")
        config = DEEPSEEK_CONFIG.copy()
        config["output_file"] = os.path.join(args.output_dir, config["output_file"])
        deepseek = DeepseekInference(config)
        deepseek.batch_inference(args.image_folder)
    
    if args.model in ["qwen", "all"]:
        print("\n=== Running Qwen Inference ===")
        config = QWEN_CONFIG.copy()
        config["output_file"] = os.path.join(args.output_dir, config["output_file"])
        qwen = QwenInference(config)
        qwen.batch_inference(args.image_folder)

if __name__ == "__main__":
    main() 