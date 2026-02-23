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
# import sys
# import hydra
# from pathlib import Path
# from omegaconf import DictConfig, OmegaConf
# from hydra import main as hydra_main
# from hydra.core.hydra_config import HydraConfig



# # Add repo root to Python path to find other modules
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(script_dir)
# repo_root = os.path.dirname(src_dir)
# if repo_root not in sys.path:
#     sys.path.insert(0, repo_root)

# # Import the function that does the actual work
# from src.data_preparation.prepare_usecase_dataset import prepare_datasets



# @hydra.main(version_base=None, config_path="../../conf", config_name="config1/default")
# def main(cfg: DictConfig) -> None:
#     """Executes only the data preparation pipeline stage."""



#     # OmegaConf.set_struct(cfg, False)  # Disable struct mode
#     # # Now regular overrides work



#     print("\n" + "="*80)
#     print("🚀 STANDALONE PIPELINE: DATA PREPARATION")
#     print("="*80)
#     print(f"\n📋 Data Config:\n{OmegaConf.to_yaml(cfg)}")
#     # //////////////////////////
#     hydra_cfg = HydraConfig.get()
#     config_name = hydra_cfg.job.config_name
#     config_key = config_name.split('/')[0] 
#     print(config_key)

#     cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
#     print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")
#     # ///////////////////////////
    
#     print(f"\n📋 data prep config:\n{OmegaConf.to_yaml(cfg.data_prep)}")
    
#     if cfg.pipeline.stages.data_preparation:
#         print("\n" + "="*80)
#         print("STAGE 1: DATA PREPARATION")
#         print("="*80)

#         try:
#             print("\n▶️  Starting data preparation...")
#             print("\n📌 Resolved data preparation paths:")
#             print(f"   CSVs       : {cfg.data_prep.input_csv_paths}")
#             print(f"   Images dir : {cfg.data_prep.image_base_dir}")
#             print(f"   Output dir : {cfg.data_prep.output_dir}")


#             print("\n🔍 Validating inputs...")
#             for csv_path in cfg.data_prep.input_csv_paths:
#                 if not os.path.exists(csv_path):
#                     print(f"❌ CSV not found: {csv_path}")
#                     sys.exit(1) 
#                 print(f"   ✅ CSV: {csv_path}")

#             if not os.path.exists(cfg.data_prep.image_base_dir):
#                 raise RuntimeError(
#                     f"Image directory not found: {cfg.data_prep.image_base_dir}"
#                 )
#             print(f"   ✅ Image directory: {cfg.data_prep.image_base_dir}")
            
#             os.makedirs(cfg.data_prep.output_dir, exist_ok=True)
#             print(f"   ✅ Output: {cfg.data_prep.output_dir}")

#             print("\n▶️  Starting data preparation...")
#             # Run the core data preparation function
#             prepare_datasets(cfg.data_prep)
#             for p in Path(cfg.data_prep.output_dir).rglob("*"):
#                 print("   ", p)
#             print("\n✅ Data preparation completed!")
#             print(f"\n📂 Output files:")
#             print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_train_file)}")
#             print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_val_file)}")
#             print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_test_file)}")
            
#             print("\n✅ Data preparation completed successfully!")

#         except Exception as e:
#             print(f"\n❌ Data preparation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)
#     else:
#         print("\n⏭️  Data preparation stage is disabled (pipeline.stages.data_preparation: false)")

# if __name__ == "__main__":
#     main()








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
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig



# Add repo root to Python path to find other modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the function that does the actual work
from src.data_preparation.prepare_usecase_dataset import prepare_datasets



@hydra.main(version_base=None, config_path="../../conf", config_name="config1/default")
def main(cfg: DictConfig) -> None:
    """Executes only the data preparation pipeline stage."""



    # OmegaConf.set_struct(cfg, False)  # Disable struct mode
    # # Now regular overrides work



    print("\n" + "="*80)
    print("🚀 STANDALONE PIPELINE: DATA PREPARATION")
    print("="*80)
    print(f"\n📋 Data Config:\n{OmegaConf.to_yaml(cfg)}")
    # //////////////////////////
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    config_key = config_name.split('/')[0] 
    print(config_key)

    cfg=cfg[config_key] if hasattr(cfg, config_key) else cfg
    print(f"\n📋 Config:\n{OmegaConf.to_yaml(cfg)}")
    # ///////////////////////////
    
    print(f"\n📋 data prep config:\n{OmegaConf.to_yaml(cfg.data_prep)}")
    
    if cfg.pipeline.stages.data_preparation:
        print("\n" + "="*80)
        print("STAGE 1: DATA PREPARATION")
        print("="*80)

        try:
            print("\n▶️  Starting data preparation...")
            print("\n📌 Resolved data preparation paths:")
            print(f"   CSVs       : {cfg.data_prep.input_csv_paths}")
            print(f"   Images dir : {cfg.data_prep.image_base_dir}")
            print(f"   Output dir : {cfg.data_prep.output_dir}")


            print("\n🔍 Validating inputs...")
            for csv_path in cfg.data_prep.input_csv_paths:
                if not os.path.exists(csv_path):
                    print(f"❌ CSV not found: {csv_path}")
                    sys.exit(1) 
                print(f"   ✅ CSV: {csv_path}")

            if not os.path.exists(cfg.data_prep.image_base_dir):
                raise RuntimeError(
                    f"Image directory not found: {cfg.data_prep.image_base_dir}"
                )
            print(f"   ✅ Image directory: {cfg.data_prep.image_base_dir}")
            
            os.makedirs(cfg.data_prep.output_dir, exist_ok=True)
            print(f"   ✅ Output: {cfg.data_prep.output_dir}")

            print("\n▶️  Starting data preparation...")
            # Run the core data preparation function
            prepare_datasets(cfg.data_prep)
            for p in Path(cfg.data_prep.output_dir).rglob("*"):
                print("   ", p)
            print("\n✅ Data preparation completed!")
            print(f"\n📂 Output files:")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_train_file)}")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_val_file)}")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_test_file)}")
            
            print("\n✅ Data preparation completed successfully!")

        except Exception as e:
            print(f"\n❌ Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⏭️  Data preparation stage is disabled (pipeline.stages.data_preparation: false)")

if __name__ == "__main__":
    main()








