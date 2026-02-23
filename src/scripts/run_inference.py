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
import sys
import importlib

import hydra
from omegaconf import DictConfig,OmegaConf
from pathlib import Path
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

# Add the current working directory to the python path
sys.path.append(os.getcwd())



def get_latest_checkpoint(exported_root: str) -> Path:
    """
    exported_root/
      ├── v1-xxxx/
      │   ├── checkpoint-100/
      ├── v2-xxxx/
      │   ├── checkpoint-200/
    """
    exported_root = Path(exported_root)

    if not exported_root.exists():
        raise FileNotFoundError(f"Exported root not found: {exported_root}")

    versions = sorted(
        [d for d in exported_root.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda x: x.name
    )

    if not versions:
        raise FileNotFoundError(f"No version folders found in {exported_root}")

    latest_version = versions[-1]
    print(f"📁 Using latest version: {latest_version.name}")

    checkpoints = sorted(
        [d for d in latest_version.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {latest_version}")

    latest_ckpt = checkpoints[-1]
    print(f"🔍 Using latest checkpoint: {latest_ckpt.name}")

    return latest_ckpt



def load_prompt(module_path, variable_name):
    """Dynamically load a prompt from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, variable_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading prompt: {e}")
        return None



@hydra.main(version_base=None,config_path="../../conf", config_name="config1/default")
def main(cfg: DictConfig):


    print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")

    print("\n" + "="*80)
    print("🚀 STANDALONE PIPELINE: INFERENCE")
    print("="*80)


    # //////////////////////////
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    config_key = config_name.split('/')[0] 
    print(config_key)

    cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
    print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")

    # ///////////////////////////
    if cfg.pipeline.stages.inference:

        
        print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")


        print(f"🔍 Searching for model in: {cfg.inference.paths.exported_model_root}")

        

        latest_ckpt = get_latest_checkpoint(str(cfg.inference.paths.exported_model_root))
        version_dir = latest_ckpt.parent.name        # v4-20260201-202214
        checkpoint_dir = latest_ckpt.name            # checkpoint-159

        # 🔑 Set model_name dynamically
        cfg.inference.DEEPSEEK_CONFIG.model_name = str(latest_ckpt)
        cfg.inference.QWEN_CONFIG.model_name = str(latest_ckpt)

        cfg.inference.paths.inference_output_dir = (
            Path(cfg.inference.paths.inference_output_dir)  
            / version_dir
        )

        os.makedirs(cfg.inference.paths.inference_output_dir, exist_ok=True)


        if cfg.model in ["deepseek", "all"]:
        
            cfg.inference.DEEPSEEK_CONFIG.output_file = str(
                cfg.inference.paths.inference_output_dir / "frozen_vit_frozen_llm_90_output.json"
            )
            print(f"📦 Inference model path: {cfg.inference.DEEPSEEK_CONFIG.model_name}")
            print(f"📦 Inference output file: {cfg.inference.DEEPSEEK_CONFIG.output_file}")
    
    

            sys.path.insert(0, '/')
            # sys.path.insert(0, '/deepseek_libs')

            from src.inference.local_lvms.deepseek_inference import DeepseekInference

            # ////////////////////
            print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg.inference)}")
            # ////////////////////


            print("\n=== Running Deepseek Inference ===")
            config = dict(cfg.inference.DEEPSEEK_CONFIG)
            config["sampling_params"] = dict(cfg.inference.SAMPLING_PARAMS)

            # Override prompt if specified in arguments
            if cfg.inference.DEEPSEEK_CONFIG.prompt_module and cfg.inference.DEEPSEEK_CONFIG.prompt_variable:
                config["prompt_module"] = cfg.inference.DEEPSEEK_CONFIG.prompt_module
                config["prompt_variable"] = cfg.inference.DEEPSEEK_CONFIG.prompt_variable

            # Load the prompt
            prompt = load_prompt(config["prompt_module"], config["prompt_variable"])
            if prompt:
                config["prompt"] = prompt
                deepseek = DeepseekInference(config)
                deepseek.batch_inference(cfg.inference.paths.image_folder)
            else:
                print("Failed to load prompt for Deepseek model. Skipping.")


        if cfg.model in ["qwen", "all"]:
        
            cfg.inference.QWEN_CONFIG.output_file = str(
                cfg.inference.paths.inference_output_dir / "frozen_vit_frozen_llm_90_output.json"
            )
            print(f"📦 Inference model path: {cfg.inference.QWEN_CONFIG.model_name}")
            print(f"📦 Inference output file: {cfg.inference.QWEN_CONFIG.output_file}")
    
    

            sys.path.insert(0, '/')
            # sys.path.insert(0, '/deepseek_libs')

            from src.inference.local_lvms.qwen_inference import QwenInference

        # ////////////////////
            print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg.inference)}")
        # ////////////////////


            print("\n=== Running Qwen Inference ===")
            config = dict(cfg.inference.QWEN_CONFIG)
            config["sampling_params"] = dict(cfg.inference.SAMPLING_PARAMS)
            # config["output_file"] = config.get("output_file") or os.path.join(cfg.inference.QWEN_CONFIG.output_file, os.path.basename(config["output_file"]))

            # Override prompt if specified in arguments
            if cfg.inference.QWEN_CONFIG.prompt_module and cfg.inference.QWEN_CONFIG.prompt_variable:
                config["prompt_module"] = cfg.inference.QWEN_CONFIG.prompt_module
                config["prompt_variable"] = cfg.inference.QWEN_CONFIG.prompt_variable

            # Load the prompt
            prompt = load_prompt(config["prompt_module"], config["prompt_variable"])
            if prompt:
                config["prompt"] = prompt
                qwen = QwenInference(config)
                qwen.batch_inference(cfg.inference.paths.image_folder)
            else:
                print("Failed to load prompt for Qwen model. Skipping.")

    else:
        print("Inference stage is disabled in the config.")


if __name__ == "__main__":
    main()
 