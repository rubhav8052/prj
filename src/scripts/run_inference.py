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
import json   
import mlflow

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

    checkpoints = sorted(
        [d for d in latest_version.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {latest_version}")

    latest_ckpt = checkpoints[-1]

    return latest_ckpt



def load_prompt(module_path, variable_name):
    """Dynamically load a prompt from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, variable_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading prompt: {e}")
        return None

def log_inference_metrics(output_file: str, model_prefix: str) -> None:
    """Read output JSON and log class distribution metrics to MLflow."""
    if not os.path.exists(output_file):
        print(f" Output file not found for metrics: {output_file}")
        return

    with open(output_file) as f:
        results = json.load(f)

    n_total  = len(results)
    n_failed = sum(1 for r in results if not r.get("output") or r.get("output") == "")
    n_success = n_total - n_failed
    success_rate = round((n_success / n_total) * 100, 2) if n_total > 0 else 0

    class_counts = {}
    for r in results:
        label = r.get("output", "").strip()
        if label:
            class_counts[label] = class_counts.get(label, 0) + 1

    # Log summary metrics
    mlflow.log_metrics({
        "n_total_images":  n_total,
        "n_success":       n_success,
        "n_failed":        n_failed,
        "success_rate":    success_rate,
    })

    # Log per class count + percent
    for cls, count in class_counts.items():
        safe_cls = cls.replace(" ", "_")  
        mlflow.log_metrics({
            f"{safe_cls}.count":   count,
            f"{safe_cls}.percent": round((count / n_total) * 100, 2),
        })

    print(f"{model_prefix} results: total={n_total} | success={n_success} | failed={n_failed}")
    print(f" Class distribution: {class_counts}")



@hydra.main(version_base=None,config_path="../../conf", config_name="config1/default")
def main(cfg: DictConfig):


    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

    print("\n" + "="*80)
    print("STANDALONE PIPELINE: INFERENCE")
    print("="*80)


    # //////////////////////////
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    config_key = config_name.split('/')[0] 
    print(config_key)

    cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

    # ///////////////////////////
    
    experiment_name = os.environ.get('AZUREML_ROOT_RUN_ID') 

    mlflow.set_tag("mlflow.runName", f"inference-{config_name.replace('/', '-')}")
    mlflow.set_tags({
        "stage":       "inference",
        "stage_enabled": cfg.pipeline.stages.inference,
        "config_name": config_name,
        "config_key":  config_key,
        "model":       cfg.get("model", ""),
        "experiment_name": experiment_name
    })

    if cfg.pipeline.stages.inference:
        try:
            print(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")
            print(f"Searching for model in: {cfg.inference.paths.exported_model_root}")

            

            latest_ckpt = get_latest_checkpoint(str(cfg.inference.paths.exported_model_root))
            version_dir = latest_ckpt.parent.name        # v4-20260201-202214
            checkpoint_dir = latest_ckpt.name            # checkpoint-159

            # Set model_name dynamically
            cfg.inference.DEEPSEEK_CONFIG.model_name = str(latest_ckpt)
            cfg.inference.QWEN_CONFIG.model_name = str(latest_ckpt)

            cfg.inference.paths.inference_output_dir = (
                Path(cfg.inference.paths.inference_output_dir)  
                / version_dir
            )

            os.makedirs(cfg.inference.paths.inference_output_dir, exist_ok=True)

            # log inference params
            mlflow.log_params({
                "inference.model":           cfg.get("model", ""),
                "inference.model_path":      str(latest_ckpt),
                "inference.version_dir":     version_dir,
                "inference.checkpoint":      checkpoint_dir,
                "inference.max_model_len":   cfg.inference.DEEPSEEK_CONFIG.get("max_model_len", "") if cfg.model in ["deepseek", "all"] else cfg.inference.QWEN_CONFIG.get("max_model_len", ""),
                "inference.batch_size":      cfg.inference.DEEPSEEK_CONFIG.get("batch_size", "") if cfg.model in ["deepseek", "all"] else cfg.inference.QWEN_CONFIG.get("batch_size", ""),
                "inference.temperature":     cfg.inference.SAMPLING_PARAMS.get("temperature", ""),
                "inference.top_p":           cfg.inference.SAMPLING_PARAMS.get("top_p", ""),
                "inference.max_tokens":      cfg.inference.SAMPLING_PARAMS.get("max_tokens", ""),
                "inference.repetition_penalty": cfg.inference.SAMPLING_PARAMS.get("repetition_penalty", ""),
            })


            if cfg.model in ["deepseek", "all"]:
            
                cfg.inference.DEEPSEEK_CONFIG.output_file = str(
                    cfg.inference.paths.inference_output_dir / "frozen_vit_frozen_llm_90_output.json"
                )
                print(f" Inference model path: {cfg.inference.DEEPSEEK_CONFIG.model_name}")
                print(f" Inference output file: {cfg.inference.DEEPSEEK_CONFIG.output_file}")
        
        

                sys.path.insert(0, '/')
                # sys.path.insert(0, '/deepseek_libs')

                from src.inference.local_lvms.deepseek_inference import DeepseekInference

                # ////////////////////
                print(f"\nFull config:\n{OmegaConf.to_yaml(cfg.inference)}")
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
                    log_inference_metrics(cfg.inference.DEEPSEEK_CONFIG.output_file, "inference.deepseek")
                    mlflow.set_tag("inference.deepseek.status", "SUCCESS")
                else:
                    print("Failed to load prompt for Deepseek model. Skipping.")
                    mlflow.set_tag("inference.deepseek.status", "SKIPPED")


            if cfg.model in ["qwen", "all"]:
            
                cfg.inference.QWEN_CONFIG.output_file = str(
                    cfg.inference.paths.inference_output_dir / "frozen_vit_frozen_llm_90_output.json"
                )
                print(f"Inference model path: {cfg.inference.QWEN_CONFIG.model_name}")
                print(f"Inference output file: {cfg.inference.QWEN_CONFIG.output_file}")
        
        

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
                    log_inference_metrics(cfg.inference.QWEN_CONFIG.output_file, "inference.qwen")
                    mlflow.set_tag("inference.qwen.status", "SUCCESS")
                else:
                    print("Failed to load prompt for Qwen model. Skipping.")
                    mlflow.set_tag("inference.qwen.status", "SKIPPED")

            lineage = {
                "inputs": {
                    "model_path":   str(latest_ckpt),
                    "version_dir":  version_dir,
                    "checkpoint":   checkpoint_dir,
                    "image_folder": str(cfg.inference.paths.image_folder),
                },
                "outputs": {
                    "inference_output_dir": str(cfg.inference.paths.inference_output_dir),
                    "output_file":          cfg.inference.DEEPSEEK_CONFIG.output_file if cfg.model in ["deepseek", "all"] else cfg.inference.QWEN_CONFIG.output_file,
                },
                "config": {
                    "model":                cfg.get("model", ""),
                    "temperature":          cfg.inference.SAMPLING_PARAMS.get("temperature", ""),
                    "top_p":                cfg.inference.SAMPLING_PARAMS.get("top_p", ""),
                    "max_tokens":           cfg.inference.SAMPLING_PARAMS.get("max_tokens", ""),
                    "repetition_penalty":   cfg.inference.SAMPLING_PARAMS.get("repetition_penalty", ""),
                }
            }
            lineage_path = Path(cfg.inference.paths.inference_output_dir) / "lineage"
            lineage_path.mkdir(parents=True, exist_ok=True)
            with open(lineage_path / "lineage.json", "w") as f:
                json.dump(lineage, f, indent=2)
            print(f"Lineage written to: {lineage_path / 'lineage.json'}")

            # ← ADDED: write resolved_config.yaml to output dir ───────────────
            config_path = Path(cfg.inference.paths.inference_output_dir) / "config"
            config_path.mkdir(parents=True, exist_ok=True)
            with open(config_path / "resolved_config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            print(f"Config written to: {config_path / 'resolved_config.yaml'}")

            mlflow.set_tag("status", "SUCCESS")     
        
        except Exception as e:  # ← ADDED
            mlflow.set_tag("status", "FAILED")
            print(f"\n Inference failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Inference stage is disabled in the config.")


if __name__ == "__main__":
    main()
 