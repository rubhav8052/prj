# # ==============================================================================
# #  C O P Y R I G H T
# # ------------------------------------------------------------------------------
# #  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# #
# #  The reproduction, distribution and utilization of this file as
# #  well as the communication of its contents to others without express
# #  authorization is prohibited. Offenders will be held liable for the
# #  payment of damages. All rights reserved in the event of the grant
# #  of a patent, utility model or design.
# # ==============================================================================


# import os
# import subprocess
# import sys
# from pathlib import Path

# import hydra
# from hydra import compose, initialize
# from omegaconf import DictConfig, OmegaConf
# from omegaconf import ListConfig
# from hydra import main as hydra_main
# from hydra.core.hydra_config import HydraConfig


# def get_latest_lora_checkpoint(output_dir: str, lora_paths=None) -> list:
#     """
#     Auto-detect latest LoRA checkpoint if lora_paths is None.
    
#     Handles nested structure:
#     output_dir/
#       ├── v1-timestamp/
#       │   ├── checkpoint-100/
#       │   ├── checkpoint-200/
#       ├── v2-timestamp/
#       │   ├── checkpoint-150/
    
#     Args:
#         output_dir: Training output directory where checkpoints are saved
#         lora_paths: If provided, return as-is; if None, auto-detect
        
#     Returns:
#         List of LoRA checkpoint paths
#     """
#     if lora_paths is not None:
#         if isinstance(lora_paths, str):
#             return [lora_paths]
#         return lora_paths
    
#     # Auto-detect latest checkpoint
#     output_path = Path(output_dir)
    
#     if not output_path.exists():
#         raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
#     # Find all version folders (v*-timestamp format)
#     version_folders = sorted(
#         [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('v')],
#         key=lambda x: x.name  
#         # Sort by version name (includes timestamp)
#     )
    
#     if not version_folders:
#         raise FileNotFoundError(f"No version folders (v*-timestamp) found in {output_dir}")
    
#     # Get the latest version folder
#     latest_version = version_folders[-1]
#     print(f"📁 Found version folders: {[v.name for v in version_folders]}")
#     print(f"📁 Using latest version: {latest_version.name}")
    
#     # Find all checkpoints in the latest version folder
#     checkpoints = sorted(
#         [d for d in latest_version.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
#         key=lambda x: int(x.name.split('-')[1])
#     )
    
#     if not checkpoints:
#         raise FileNotFoundError(f"No checkpoints found in {latest_version}")
    
#     # Get the latest checkpoint
#     latest_checkpoint = str(checkpoints[-1])
#     print(f"🔍 Found checkpoints: {[c.name for c in checkpoints]}")
#     print(f"🔍 Auto-detected latest LoRA checkpoint: {latest_checkpoint}")
    
#     return [latest_checkpoint]



# # Add repo root to Python path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(script_dir)
# repo_root = os.path.dirname(src_dir)
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)



# from src.data_preparation.prepare_usecase_dataset import prepare_datasets
# from src.utils.file_utils import set_environment_variables


# @hydra.main(version_base=None, config_path="../../conf", config_name="config1/default")
# def main(cfg: DictConfig) -> None:
    
#     """Execute full pipeline: data preparation + training + export."""
    


#     print("\n" + "="*80)
#     print("🚀 FULL PIPELINE: DATA PREP + TRAINING + EXPORT")
#     print("="*80)
#     print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
#     print("-----------")
    
#     # ////////////////////////////////////////////////////

#     hydra_cfg = HydraConfig.get()
#     config_name = hydra_cfg.job.config_name
#     config_key = config_name.split('/')[0] 
#     print(config_key)

#     cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
#     print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")
    
#     # ////////////////////////////////////////////////////


#     print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")

#     # =====================================================================
#     # STAGE 2: TRAINING
#     # =====================================================================

#     if cfg.pipeline.stages.training:
#     # if False:
#         try:
#             import deepspeed  # noqa: F401
#         except ImportError:
#             env = {**os.environ, "DS_BUILD_OPS": "0", "DS_BUILD_AIO": "0", "DS_BUILD_SPARSE_ATTN": "0"}
#             if "CUDA_HOME" not in env and os.path.exists("/azureml-envs/mtl_onnx"):
#                 env["CUDA_HOME"] = "/azureml-envs/mtl_onnx"
#             try:
#                 subprocess.check_call(
#                     [sys.executable, "-m", "pip", "install", "--no-build-isolation", "deepspeed==0.18.4"],
#                     env=env
#                 )
#             except subprocess.CalledProcessError as e:
#                 print(f"⚠️ Skipping DeepSpeed install (non-fatal): {e}")

#         print("\n" + "="*80)
#         print("STAGE 2: TRAINING")
#         print("="*80)
        
#         try:
            
#             if "training" not in cfg:
#                 print("\n❌ ERROR: 'training' not found in config!")
#                 print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
#                 sys.exit(1)
        


            
#             print("\n🔍 Validating training inputs...")
            
#             train_dataset_paths = cfg.training.get('train_dataset_paths', [])
#             val_dataset_paths = cfg.training.get('val_dataset_paths', [])
            
#             if isinstance(train_dataset_paths, str):
#                 train_dataset_paths = [train_dataset_paths]
#             if isinstance(val_dataset_paths, str):
#                 val_dataset_paths = [val_dataset_paths]
            

            
#             os.makedirs(cfg.training.output_dir, exist_ok=True)
#             print(f"   ✅ Output: {cfg.training.output_dir}")
            
#             # Add swift libs path if specified
#             swift_libs_path = cfg.training.get('swift_libs_path', None)
#             if swift_libs_path:
#                 if os.path.exists(swift_libs_path):
#                     print(f"\n➕ Adding Swift libs path to sys.path: {swift_libs_path}")
#                     if swift_libs_path not in sys.path:
#                         sys.path.insert(0, swift_libs_path)
#                 else:
#                     print(f"\n⚠️  Swift libs path not found: {swift_libs_path}")
            

#             # Create images symlink if needed
#             if train_dataset_paths:

#                 path_to_check = train_dataset_paths[0] if isinstance(train_dataset_paths, (list, ListConfig)) else train_dataset_paths
                
#                 dataset_dir = os.path.dirname(path_to_check)
#                 images_src = os.path.join(dataset_dir, 'images')
                
#                 if dataset_dir and not os.path.exists('images') and os.path.exists(images_src):
#                     print(f"\n🔗 Creating images symlink...")
#                     try:
#                         os.symlink(images_src, 'images')
#                         print(f"   ✅ Symlink created: images -> {images_src}")
#                     except OSError as e:
#                         print(f"   ⚠️ Symlink failed: {e}. (This can happen if 'images' already exists)")


#             # Set environment variables from config
#             if "env_vars" in cfg.training:
#                 print("\n🔧 Setting training environment variables...")

#                 set_environment_variables(cfg.training.env_vars)

#             # Import and run training
#             print("\n📦 Importing Swift trainer module...")
#             try:
#                 from src.training.swift_trainer import run_swift_sft
#             except ModuleNotFoundError as e:
#                 print(f"\n❌ Failed to import swift: {e}")
#                 sys.exit(1)
            
#             print("\n▶️  Starting Swift SFT training...")
#             run_swift_sft(cfg)
            
#             print("\n✅ Training completed!")
            
#         except Exception as e:
#             print(f"\n❌ Training failed: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)
#     else:
#         print("\n⏭️  Training stage is disabled (pipeline.stages.training: false)")

#     # =====================================================================
#     # STAGE 3: EXPORT & QUANTIZATION
#     # =====================================================================


#     if cfg.pipeline.stages.training:

#         print("\n" + "="*80)
#         print("STAGE 3: EXPORT & QUANTIZATION")
#         print("="*80)
        
#         try:

#             if "export" not in cfg.training:
#                 print("\n❌ ERROR: 'export' not found in config!")
#                 print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
#                 sys.exit(1)

#             export_cfg = cfg.training.export
            
#             # 🔑 AUTO-DETECT LORA PATHS if None
#             if export_cfg.lora_paths is None:
                
#                 print("\n🔍 Auto-detecting LoRA checkpoint from training output...")
#                 export_cfg.lora_paths = get_latest_lora_checkpoint(
#                     cfg.training.output_dir,  # Use training output_dir
#                     # lora_paths=None
#                 )
#                 print(f"✅ LoRA paths resolved to: {export_cfg.lora_paths}")

#             if export_cfg.output_dir :

#                 latest_ckpt = Path(export_cfg.lora_paths[0])
#                 version_dir = latest_ckpt.parent.name      # v4-20260201-202214
#                 checkpoint_name = latest_ckpt.name          # checkpoint-159


#                 export_cfg.output_dir = str(
#                     Path(export_cfg.output_dir)
#                     / version_dir 
#                     / checkpoint_name
#                 )

#                 export_cfg.output_dir = str(export_cfg.output_dir)

#                 print(f"📦 Export output_dir set to: {export_cfg.output_dir}")
#             else:
#                 print("❌ Error: No LoRA checkpoints found. Cannot proceed with Export.")
#                 sys.exit(1)
                
#             print("\n📋 Export configuration:")
#             print(OmegaConf.to_yaml(export_cfg))
            
#             # Create images symlink if needed
#             if export_cfg.dataset_path:
#                 d_path = export_cfg.dataset_path[0] if isinstance(export_cfg.dataset_path, (list, ListConfig)) else export_cfg.dataset_path
#                 dataset_dir = os.path.dirname(d_path)
#                 images_src = os.path.join(dataset_dir, 'images')
                
#                 if not os.path.exists('images') and os.path.exists(images_src):
#                     print(f"\n🔗 Creating images symlink for Export...")
#                     try:
#                         os.symlink(images_src, 'images')
#                     except OSError:
#                         pass # Already exists
#                     print(f"   ✅ Symlink created: images -> {images_src}")
            
#             # Add swift_libs_path to sys.path if provided
#             if export_cfg.swift_libs_path and export_cfg.swift_libs_path not in sys.path:
#                 print(f"\n📦 Adding Swift libs path to sys.path: {export_cfg.swift_libs_path}")
#                 sys.path.insert(0, export_cfg.swift_libs_path)
            




#             if "env_vars" in cfg.training:  
#                 set_environment_variables(cfg.training.env_vars)
            

#             print("\n▶️  Starting Swift model export...")
#             from src.inference.local_lvms.swift_inference.exporter import run_swift_export
            
#             run_swift_export(export_cfg)
            
#             print("\n✅ Export completed successfully!")
            
#         except Exception as e:
#             print(f"\n❌ Export failed: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)
#     else:
#         print("\n⏭️  Export stage is disabled (pipeline.stages.export: false)")
    

# if __name__ == "__main__":
#     main()







import json

import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig

import mlflow


# Add repo root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── MLflow logging helpers ────────────────────────────────────────────────────
from src.ml_flow.train_exp import (
    log_training_start,
    log_training_params,
    log_training_checkpoints,
    log_export_params,
    log_export_metrics,
    log_training_lineage,
    log_training_status,
)




# def count_lines(path) -> int:
#     """Count lines in a JSONL file = number of samples."""
#     return sum(1 for _ in open(path)) if os.path.exists(str(path)) else 0

from src.data_preparation.prepare_usecase_dataset import prepare_datasets
from src.utils.file_utils import set_environment_variables
def get_latest_lora_checkpoint(output_dir: str, lora_paths=None) -> list:
    """
    Auto-detect latest LoRA checkpoint if lora_paths is None.
    
    Handles nested structure:
    output_dir/
      ├── v1-timestamp/
      │   ├── checkpoint-100/
      │   ├── checkpoint-200/
      ├── v2-timestamp/
      │   ├── checkpoint-150/
    
    Args:
        output_dir: Training output directory where checkpoints are saved
        lora_paths: If provided, return as-is; if None, auto-detect
        
    Returns:
        List of LoRA checkpoint paths
    """
    if lora_paths is not None:
        if isinstance(lora_paths, str):
            return [lora_paths]
        return lora_paths
    
    # Auto-detect latest checkpoint
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Find all version folders (v*-timestamp format)
    version_folders = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('v')],
        key=lambda x: x.name  
        # Sort by version name (includes timestamp)
    )
    
    if not version_folders:
        raise FileNotFoundError(f"No version folders (v*-timestamp) found in {output_dir}")
    
    # Get the latest version folder
    latest_version = version_folders[-1]
    print(f"📁 Found version folders: {[v.name for v in version_folders]}")
    print(f"📁 Using latest version: {latest_version.name}")
    
    # Find all checkpoints in the latest version folder
    checkpoints = sorted(
        [d for d in latest_version.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[1])
    )
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_version}")
    
    # Get the latest checkpoint
    latest_checkpoint = str(checkpoints[-1])
    print(f"🔍 Found checkpoints: {[c.name for c in checkpoints]}")
    print(f"🔍 Auto-detected latest LoRA checkpoint: {latest_checkpoint}")
    
    return [latest_checkpoint]



@hydra.main(version_base=None, config_path="../../conf", config_name="config1/default")
def main(cfg: DictConfig) -> None:
    
    """Execute full pipeline: data preparation + training + export."""
    

    
    print("\n" + "="*80)
    print(" TRAINING + EXPORT")
    print("="*80)
    print(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")
    print("-----------")
    
    # ////////////////////////////////////////////////////

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    config_key = config_name.split('/')[0] 
    print(config_key)

    cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
    print(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # ////////////////////////////////////////////////////
    print(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")

    experiment_name = os.environ.get('AZUREML_ROOT_RUN_ID') 
    
    # # mlflow.set_experiment("attribute-labeling-pipeline")
    # mlflow.set_tag("mlflow.runName", f"train_export-{config_name.replace('/', '-')}[{experiment_name}]")
    # # mlflow.set_tag("mlflow.runName", f"train_export-{config_name.replace('/', '-')}")
    # # with mlflow.start_run(run_name=f"train_export-{config_name.replace('/', '-')}"):
    # mlflow.set_tags({
    # "stage":       "train_export",
    # "stage_enabled": cfg.pipeline.stages.training,
    # "config_name": config_name,
    # "config_key":  config_key,
    # "model_id":    cfg.training.model_id,
    # "experiment_name": experiment_name
    # })

    # MLflow: run name + tags
    log_training_start(cfg, config_name, config_key, experiment_name)

    # =====================================================================
    # STAGE 2: TRAINING
    # =====================================================================

    if cfg.pipeline.stages.training:
    # if False:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            env = {**os.environ, "DS_BUILD_OPS": "0", "DS_BUILD_AIO": "0", "DS_BUILD_SPARSE_ATTN": "0"}
            if "CUDA_HOME" not in env and os.path.exists("/azureml-envs/mtl_onnx"):
                env["CUDA_HOME"] = "/azureml-envs/mtl_onnx"
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-build-isolation", "deepspeed==0.18.4"],
                    env=env
                )
            except subprocess.CalledProcessError as e:
                print(f"Skipping DeepSpeed install (non-fatal): {e}")

        print("\n" + "="*80)
        print("STAGE 2: TRAINING")
        print("="*80)
        
        try:
            
            if "training" not in cfg:
                print("\nERROR: 'training' not found in config!")
                print(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")
                sys.exit(1)
        
            # --- MLflow: log tags ---
            # mlflow.set_tags({
            #     "stage": "training",
            #     "config_name": config_name,
            #     "config_key":  config_key,
            #     "model_id":    cfg.training.model_id,
            # })

            
            print("\n🔍 Validating training inputs...")
            
            train_dataset_paths = cfg.training.get('train_dataset_paths', [])
            val_dataset_paths = cfg.training.get('val_dataset_paths', [])
            
            if isinstance(train_dataset_paths, str):
                train_dataset_paths = [train_dataset_paths]
            if isinstance(val_dataset_paths, str):
                val_dataset_paths = [val_dataset_paths]
            

            
            os.makedirs(cfg.training.output_dir, exist_ok=True)
            print(f"Output: {cfg.training.output_dir}")
            
            # ← ADDED: log training params
            # mlflow.log_params({
            #     "train.model_id":                    cfg.training.model_id,
            #     "train.train_type":                  cfg.training.get("train_type", ""),
            #     "train.lora_rank":                   cfg.training.get("lora_rank", ""),
            #     "train.freeze_vit":                  cfg.training.get("freeze_vit", ""),
            #     "train.freeze_parameters_ratio":     cfg.training.get("freeze_parameters_ratio", ""),
            #     "train.num_train_epochs":            cfg.training.get("num_train_epochs", ""),
            #     "train.batch_size":                  cfg.training.get("batch_size", ""),
            #     "train.gradient_accumulation_steps": cfg.training.get("gradient_accumulation_steps", ""),
            #     "train.learning_rate":               cfg.training.get("learning_rate", ""),
            #     "train.loss_type":                   cfg.training.get("loss_type", ""),
            #     "train.torch_dtype":                 cfg.training.get("torch_dtype", ""),
            #     "train.output_dir":                  cfg.training.output_dir,
            # })

            

            # train_path = train_dataset_paths[0] if isinstance(train_dataset_paths, (list, ListConfig)) else train_dataset_paths
            # val_path   = val_dataset_paths[0]   if isinstance(val_dataset_paths,   (list, ListConfig)) else val_dataset_paths

            # n_train = count_lines(train_path)
            # n_val   = count_lines(val_path)
            # # ← ADDED: derived metrics useful for comparison
            # effective_batch = cfg.training.get("batch_size", 1) * cfg.training.get("gradient_accumulation_steps", 1)
            # mlflow.log_metrics({
            #     "train.effective_batch_size": effective_batch,
            #     "train.n_train_samples":      n_train,
            #     "train.n_val_samples":        n_val,
            # })


            # ── MLflow: params + derived metrics ──────────────────────────────
            log_training_params(cfg)

            # Add swift libs path if specified
            swift_libs_path = cfg.training.get('swift_libs_path', None)
            if swift_libs_path:
                if os.path.exists(swift_libs_path):
                    print(f"\nAdding Swift libs path to sys.path: {swift_libs_path}")
                    if swift_libs_path not in sys.path:
                        sys.path.insert(0, swift_libs_path)
                else:
                    print(f"\nSwift libs path not found: {swift_libs_path}")
            

            # Create images symlink if needed
            if train_dataset_paths:

                path_to_check = train_dataset_paths[0] if isinstance(train_dataset_paths, (list, ListConfig)) else train_dataset_paths
                
                dataset_dir = os.path.dirname(path_to_check)
                images_src = os.path.join(dataset_dir, 'images')
                
                if dataset_dir and not os.path.exists('images') and os.path.exists(images_src):
                    print(f"\n🔗 Creating images symlink...")
                    try:
                        os.symlink(images_src, 'images')
                        print(f"Symlink created: images -> {images_src}")
                    except OSError as e:
                        print(f"Symlink failed: {e}. (This can happen if 'images' already exists)")


            # Set environment variables from config
            if "env_vars" in cfg.training:
                print("\nSetting training environment variables...")

                set_environment_variables(cfg.training.env_vars)

            # Import and run training
            print("\nImporting Swift trainer module...")
            try:
                from src.training.swift_trainer import run_swift_sft
            except ModuleNotFoundError as e:
                print(f"\nFailed to import swift: {e}")
                sys.exit(1)
            
            print("\nStarting Swift SFT training...")
            run_swift_sft(cfg)
            
            print("\nTraining completed!")
            

            # # ← ADDED: log checkpoint metrics after training
            # checkpoints = sorted(
            #     [d for d in Path(cfg.training.output_dir).rglob("checkpoint-*") if d.is_dir()],
            #     key=lambda x: int(x.name.split("-")[1])
            # )
            # if checkpoints:
            #     latest_ckpt_step = int(checkpoints[-1].name.split("-")[1])
            #     mlflow.log_metric("train.total_checkpoints_saved", len(checkpoints))
            #     mlflow.log_metric("train.final_checkpoint_step",   latest_ckpt_step)
            #     mlflow.set_tag("train.latest_checkpoint", str(checkpoints[-1]))

                
            #     # ── Read trainer_state.json for summary + curve metrics ───────
            #     trainer_state_path = checkpoints[-1] / "trainer_state.json"
            #     if trainer_state_path.exists():
            #         with open(trainer_state_path) as f:
            #             trainer_state = json.load(f)

            #         log_history = trainer_state.get("log_history", [])

            #         # ── Summary metrics ───────────────────────────────────────
            #         best_ckpt = trainer_state.get("best_model_checkpoint", "")
            #         mlflow.log_metrics({
            #             "train.best_loss":          trainer_state.get("best_metric", 0),
            #             "train.final_epoch":        trainer_state.get("epoch", 0),
            #             "train.actual_total_steps": trainer_state.get("global_step", 0),
            #             "train.max_steps":          trainer_state.get("max_steps", 0),
            #             "train.train_batch_size":   trainer_state.get("train_batch_size", 0),
            #         })
            #         mlflow.set_tag("train.best_checkpoint", best_ckpt)
            #         mlflow.set_tag("train.best_is_final",   str(best_ckpt == str(checkpoints[-1])))

            #         # ── Loss improvement ──────────────────────────────────────
            #         train_entries = [e for e in log_history if "loss" in e]
            #         if len(train_entries) >= 2:
            #             first_loss       = train_entries[0]["loss"]
            #             last_loss        = train_entries[-1]["loss"]
            #             loss_improvement = round(first_loss - last_loss, 4)
            #             loss_improvement_pct = round((loss_improvement / first_loss) * 100, 2) if first_loss > 0 else 0
            #             mlflow.log_metrics({
            #                 "train.first_loss":           first_loss,
            #                 "train.last_loss":            last_loss,
            #                 "train.loss_improvement":     loss_improvement,
            #                 "train.loss_improvement_pct": loss_improvement_pct,
            #             })

            #         # ── Eval loss improvement ─────────────────────────────────
            #         eval_entries = [e for e in log_history if "eval_loss" in e]
            #         if eval_entries:
            #             mlflow.log_metrics({
            #                 "train.first_eval_loss": eval_entries[0]["eval_loss"],
            #                 "train.last_eval_loss":  eval_entries[-1]["eval_loss"],
            #                 "train.best_eval_loss":  min(e["eval_loss"] for e in eval_entries),
            #             })

            #         # ── Token accuracy improvement ────────────────────────────
            #         token_entries = [e for e in log_history if "token_acc" in e]
            #         if token_entries:
            #             eval_token_entries = [e for e in log_history if "eval_token_acc" in e]
            #             mlflow.log_metrics({
            #                 "train.first_token_acc": token_entries[0]["token_acc"],
            #                 "train.last_token_acc":  token_entries[-1]["token_acc"],
            #                 "train.best_eval_token_acc": max(
            #                     e["eval_token_acc"] for e in eval_token_entries
            #                 ) if eval_token_entries else 0,
            #             })

            #         # ── Final training speed + memory ─────────────────────────
            #         if train_entries:
            #             last_entry = train_entries[-1]
            #             mlflow.log_metrics({
            #                 "train.final_train_speed_iter_per_s": last_entry.get("train_speed(iter/s)", 0),
            #                 "train.final_memory_gb":              last_entry.get("memory(GiB)", 0),
            #             })

            #         # ── Per epoch eval metrics (gives epoch-level graphs) ─────
            #         for entry in log_history:
            #             if "eval_loss" in entry:
            #                 epoch = round(entry.get("epoch", 0))
            #                 mlflow.log_metric("train.eval_loss_per_epoch",      entry["eval_loss"],            step=epoch)
            #                 mlflow.log_metric("train.eval_token_acc_per_epoch", entry.get("eval_token_acc", 0), step=epoch)

            #         print(f"Logged summary + curve metrics from trainer_state.json")
            #     else:
            #         print(f"trainer_state.json not found at: {trainer_state_path}")


            # mlflow.set_tag("train.status", "SUCCESS")
            
            # ── MLflow: checkpoint metrics ────────────────────────────────────
            log_training_checkpoints(cfg)
            log_training_status("train", "SUCCESS")

        except Exception as e:
            log_training_status("train", "FAILED")
            print(f"\nTraining failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nTraining stage is disabled (pipeline.stages.training: false)")

    # =====================================================================
    # STAGE 3: EXPORT & QUANTIZATION
    # =====================================================================


    if cfg.pipeline.stages.training:

        print("\n" + "="*80)
        print("STAGE 3: EXPORT & QUANTIZATION")
        print("="*80)
        
        try:

            if "export" not in cfg.training:
                print("\nERROR: 'export' not found in config!")
                print(f"\nFull config:\n{OmegaConf.to_yaml(cfg)}")
                sys.exit(1)

            export_cfg = cfg.training.export
            
            # 🔑 AUTO-DETECT LORA PATHS if None
            if export_cfg.lora_paths is None:
                
                print("\n🔍 Auto-detecting LoRA checkpoint from training output...")
                export_cfg.lora_paths = get_latest_lora_checkpoint(
                    cfg.training.output_dir,  # Use training output_dir
                    # lora_paths=None
                )
                print(f"LoRA paths resolved to: {export_cfg.lora_paths}")

            if export_cfg.output_dir :

                latest_ckpt = Path(export_cfg.lora_paths[0])
                version_dir = latest_ckpt.parent.name      # v4-20260201-202214
                checkpoint_name = latest_ckpt.name          # checkpoint-159


                export_cfg.output_dir = str(
                    Path(export_cfg.output_dir)
                    / version_dir 
                    / checkpoint_name
                )

                export_cfg.output_dir = str(export_cfg.output_dir)

                print(f"Export output_dir set to: {export_cfg.output_dir}")

                # mlflow.log_params({
                #     "export.model":       export_cfg.get("model", ""),
                #     "export.lora_path":   str(export_cfg.lora_paths[0]),
                #     "export.output_dir":  export_cfg.output_dir,
                #     "export.merge_lora":  export_cfg.get("merge_lora", True),
                # })
                # mlflow.set_tags({
                #     "export.version_dir":  version_dir,
                #     "export.checkpoint":   checkpoint_name,
                # })
                # mlflow.log_metric("export.checkpoint_step", int(checkpoint_name.split("-")[1]))

                # ── MLflow: export params ─────────────────────────────────────────
                log_export_params(export_cfg, version_dir, checkpoint_name)

            else:
                print("Error: No LoRA checkpoints found. Cannot proceed with Export.")
                sys.exit(1)

            print("\nExport configuration:")
            print(OmegaConf.to_yaml(export_cfg))
            



            # Create images symlink if needed
            if export_cfg.dataset_path:
                d_path = export_cfg.dataset_path[0] if isinstance(export_cfg.dataset_path, (list, ListConfig)) else export_cfg.dataset_path
                dataset_dir = os.path.dirname(d_path)
                images_src = os.path.join(dataset_dir, 'images')
                
                if not os.path.exists('images') and os.path.exists(images_src):
                    print(f"\nCreating images symlink for Export...")
                    try:
                        os.symlink(images_src, 'images')
                    except OSError:
                        pass # Already exists
                    print(f"   Symlink created: images -> {images_src}")
            
            # Add swift_libs_path to sys.path if provided
            if export_cfg.swift_libs_path and export_cfg.swift_libs_path not in sys.path:
                print(f"\nAdding Swift libs path to sys.path: {export_cfg.swift_libs_path}")
                sys.path.insert(0, export_cfg.swift_libs_path)
            




            if "env_vars" in cfg.training:  
                set_environment_variables(cfg.training.env_vars)
            

            print("\nStarting Swift model export...")
            from src.inference.local_lvms.swift_inference.exporter import run_swift_export
            
            run_swift_export(export_cfg)
            
            print("\nExport completed successfully!")

            # # ── Log exported model size on disk ───────────────────────────────
            # export_path = Path(export_cfg.output_dir)
            # if export_path.exists():
            #     export_size_gb = sum(
            #         f.stat().st_size for f in export_path.rglob("*") if f.is_file()
            #     ) / (1024 ** 3)
            #     mlflow.log_metric("export.model_size_gb", round(export_size_gb, 3))
            #     print(f"Exported model size: {export_size_gb:.3f} GB")



            # output_dir = Path(export_cfg.output_dir)
            # lineage_path = output_dir / "lineage"
            # lineage_path.mkdir(parents=True, exist_ok=True)

            # # Create a lineage dictionary to track inputs and outputs
            # lineage = {
            #     "inputs": {
            #         "model_id": cfg.training.model_id,
            #         "lora_checkpoint": str(export_cfg.lora_paths[0]),
            #         "train_datasets": OmegaConf.to_container(cfg.training.get('train_dataset_paths', []), resolve=True),
            #     },
            #     "outputs": {
            #         "export_dir": str(output_dir)
            #     }
            # }
            # # Write the lineage information to a JSON file
            # with open(lineage_path / "lineage.json", "w") as f:
            #     json.dump(lineage, f, indent=2)
            # print(f"Lineage written to: {lineage_path / 'lineage.json'}")

            # # Create a path to save the configuration
            # config_path = output_dir / "config"
            # config_path.mkdir(parents=True, exist_ok=True)

            # # Write the full, resolved Hydra configuration to a YAML file
            # with open(config_path / "resolved_config.yaml", "w") as f:
            #     f.write(OmegaConf.to_yaml(cfg))
            # print(f"Config written to: {config_path / 'resolved_config.yaml'}")
            # mlflow.set_tag("export.status", "SUCCESS")  
            
            # ── MLflow: export metrics + lineage ──────────────────────────────
            log_export_metrics(export_cfg)
            log_training_lineage(cfg, export_cfg, version_dir, checkpoint_name)
            log_training_status("export", "SUCCESS")
            
        except Exception as e:
            log_training_status("export", "FAILED")
            print(f"\nExport failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nExport stage is disabled (pipeline.stages.export: false)")
    

if __name__ == "__main__":
    main()
