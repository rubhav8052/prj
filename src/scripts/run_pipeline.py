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
import subprocess
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig


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



# Add repo root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)



from src.data_preparation.prepare_usecase_dataset import prepare_datasets
from src.utils.file_utils import set_environment_variables


@hydra.main(version_base=None, config_path="../../conf", config_name="config1/default")
def main(cfg: DictConfig) -> None:
    
    """Execute full pipeline: data preparation + training + export."""
    


    print("\n" + "="*80)
    print("🚀 FULL PIPELINE: DATA PREP + TRAINING + EXPORT")
    print("="*80)
    print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
    print("-----------")
    
    # ////////////////////////////////////////////////////

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    config_key = config_name.split('/')[0] 
    print(config_key)

    cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
    print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # ////////////////////////////////////////////////////


    print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")

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
                print(f"⚠️ Skipping DeepSpeed install (non-fatal): {e}")

        print("\n" + "="*80)
        print("STAGE 2: TRAINING")
        print("="*80)
        
        try:
            
            if "training" not in cfg:
                print("\n❌ ERROR: 'training' not found in config!")
                print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
                sys.exit(1)
        


            
            print("\n🔍 Validating training inputs...")
            
            train_dataset_paths = cfg.training.get('train_dataset_paths', [])
            val_dataset_paths = cfg.training.get('val_dataset_paths', [])
            
            if isinstance(train_dataset_paths, str):
                train_dataset_paths = [train_dataset_paths]
            if isinstance(val_dataset_paths, str):
                val_dataset_paths = [val_dataset_paths]
            

            
            os.makedirs(cfg.training.output_dir, exist_ok=True)
            print(f"   ✅ Output: {cfg.training.output_dir}")
            
            # Add swift libs path if specified
            swift_libs_path = cfg.training.get('swift_libs_path', None)
            if swift_libs_path:
                if os.path.exists(swift_libs_path):
                    print(f"\n➕ Adding Swift libs path to sys.path: {swift_libs_path}")
                    if swift_libs_path not in sys.path:
                        sys.path.insert(0, swift_libs_path)
                else:
                    print(f"\n⚠️  Swift libs path not found: {swift_libs_path}")
            

            # Create images symlink if needed
            if train_dataset_paths:

                path_to_check = train_dataset_paths[0] if isinstance(train_dataset_paths, (list, ListConfig)) else train_dataset_paths
                
                dataset_dir = os.path.dirname(path_to_check)
                images_src = os.path.join(dataset_dir, 'images')
                
                if dataset_dir and not os.path.exists('images') and os.path.exists(images_src):
                    print(f"\n🔗 Creating images symlink...")
                    try:
                        os.symlink(images_src, 'images')
                        print(f"   ✅ Symlink created: images -> {images_src}")
                    except OSError as e:
                        print(f"   ⚠️ Symlink failed: {e}. (This can happen if 'images' already exists)")


            # Set environment variables from config
            if "env_vars" in cfg.training:
                print("\n🔧 Setting training environment variables...")

                set_environment_variables(cfg.training.env_vars)

            # Import and run training
            print("\n📦 Importing Swift trainer module...")
            try:
                from src.training.swift_trainer import run_swift_sft
            except ModuleNotFoundError as e:
                print(f"\n❌ Failed to import swift: {e}")
                sys.exit(1)
            
            print("\n▶️  Starting Swift SFT training...")
            run_swift_sft(cfg)
            
            print("\n✅ Training completed!")
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⏭️  Training stage is disabled (pipeline.stages.training: false)")

    # =====================================================================
    # STAGE 3: EXPORT & QUANTIZATION
    # =====================================================================


    if cfg.pipeline.stages.training:

        print("\n" + "="*80)
        print("STAGE 3: EXPORT & QUANTIZATION")
        print("="*80)
        
        try:

            if "export" not in cfg.training:
                print("\n❌ ERROR: 'export' not found in config!")
                print(f"\n📋 Full config:\n{OmegaConf.to_yaml(cfg)}")
                sys.exit(1)

            export_cfg = cfg.training.export
            
            # 🔑 AUTO-DETECT LORA PATHS if None
            if export_cfg.lora_paths is None:
                
                print("\n🔍 Auto-detecting LoRA checkpoint from training output...")
                export_cfg.lora_paths = get_latest_lora_checkpoint(
                    cfg.training.output_dir,  # Use training output_dir
                    # lora_paths=None
                )
                print(f"✅ LoRA paths resolved to: {export_cfg.lora_paths}")

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

                print(f"📦 Export output_dir set to: {export_cfg.output_dir}")
            else:
                print("❌ Error: No LoRA checkpoints found. Cannot proceed with Export.")
                sys.exit(1)
                
            print("\n📋 Export configuration:")
            print(OmegaConf.to_yaml(export_cfg))
            
            # Create images symlink if needed
            if export_cfg.dataset_path:
                d_path = export_cfg.dataset_path[0] if isinstance(export_cfg.dataset_path, (list, ListConfig)) else export_cfg.dataset_path
                dataset_dir = os.path.dirname(d_path)
                images_src = os.path.join(dataset_dir, 'images')
                
                if not os.path.exists('images') and os.path.exists(images_src):
                    print(f"\n🔗 Creating images symlink for Export...")
                    try:
                        os.symlink(images_src, 'images')
                    except OSError:
                        pass # Already exists
                    print(f"   ✅ Symlink created: images -> {images_src}")
            
            # Add swift_libs_path to sys.path if provided
            if export_cfg.swift_libs_path and export_cfg.swift_libs_path not in sys.path:
                print(f"\n📦 Adding Swift libs path to sys.path: {export_cfg.swift_libs_path}")
                sys.path.insert(0, export_cfg.swift_libs_path)
            




            if "env_vars" in cfg.training:  
                set_environment_variables(cfg.training.env_vars)
            

            print("\n▶️  Starting Swift model export...")
            from src.inference.local_lvms.swift_inference.exporter import run_swift_export
            
            run_swift_export(export_cfg)
            
            print("\n✅ Export completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⏭️  Export stage is disabled (pipeline.stages.export: false)")
    

if __name__ == "__main__":
    main()