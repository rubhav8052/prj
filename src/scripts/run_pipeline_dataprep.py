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


# import json
# import mlflow

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


#             with mlflow.start_run():
#             # ---- Identity tags (filtering & grouping) ----
#                 mlflow.set_tags({
#                     "stage": "data_prep",
#                     "config_name": config_name,
#                     "dataset.name": "tl_inlay",
#                     "dataset.variant": config_name,
#                 })

#                 # ---- Important parameters ONLY ----
#                 mlflow.log_params({
#                     # data identity
#                     "num_csvs": len(cfg.data_prep.input_csv_paths),
#                     "index_col": cfg.data_prep.index_col,
#                     "attribute_col": cfg.data_prep.attribute_col,

#                     # split logic
#                     "test_size": cfg.data_prep.test_size,
#                     "val_size": cfg.data_prep.val_size,
#                     "random_state": cfg.data_prep.random_state,

#                     # filtering logic
#                     "min_height_width_pixel": cfg.data_prep.min_height_width_pixel,
#                     "context_percent": cfg.data_prep.context_percent,
#                 })

#                 # ---- Lineage artifact (THIS is what you were missing) ----
#                 lineage = {
#                     "inputs": {
#                         "csvs": OmegaConf.to_container(cfg.data_prep.input_csv_paths, resolve=True),
#                         "image_base_dir": cfg.data_prep.image_base_dir,
#                     },
#                     "outputs": {
#                         "data_prep_dir": cfg.data_prep.output_dir,
#                         "train_file": cfg.data_prep.attribute_train_file,
#                         "val_file": cfg.data_prep.attribute_val_file,
#                         "test_file": cfg.data_prep.attribute_test_file,
#                     },
#                 }

#                 lineage_path = Path(cfg.data_prep.output_dir) / "lineage.json"
#                 lineage_path.write_text(json.dumps(lineage, indent=2))

#                 # mlflow.log_artifact(str(lineage_path), artifact_path="lineage")
#                 # mlflow.log_dict(lineage, artifact_file="lineage/data_lineage.json")


#                 (Path(cfg.data_prep.output_dir) / "resolved_config.yaml").write_text(
#                     OmegaConf.to_yaml(cfg)
#                 )
#                 # ---- Full resolved config (single source of truth) ----
#                 # resolved_cfg_path = Path(cfg.data_prep.output_dir) / "resolved_config.yaml"
#                 # resolved_cfg_path.write_text(OmegaConf.to_yaml(cfg))
#                 # mlflow.log_artifact(str(resolved_cfg_path), artifact_path="config")
#                 # mlflow.log_text(
#                 #     OmegaConf.to_yaml(cfg),
#                 #     artifact_file="config/resolved_config.yaml"
#                 # )

#                 print("\n▶️  Starting data preparation...")
#                 # Run the core data preparation function
#                 prepare_datasets(cfg.data_prep)
#                 for p in Path(cfg.data_prep.output_dir).rglob("*"):
#                     print("   ", p)
#                 print("\n✅ Data preparation completed!")
#                 print(f"\n📂 Output files:")
#                 print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_train_file)}")
#                 print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_val_file)}")
#                 print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_test_file)}")
                
#                 print("\n✅ Data preparation completed successfully!")

#                 def count_lines(p):
#                     return sum(1 for _ in open(p)) if p.exists() else 0

#                 mlflow.log_metrics({
#                     "num_train_samples": count_lines(
#                         Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_train_file
#                     ),
#                     "num_val_samples": count_lines(
#                         Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_val_file
#                     ),
#                     "num_test_samples": count_lines(
#                         Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_test_file
#                     ),
#                 })

#         except Exception as e:
#             print(f"\n❌ Data preparation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)
#     else:
#         print("\n⏭️  Data preparation stage is disabled (pipeline.stages.data_preparation: false)")

# if __name__ == "__main__":
    
#     main()









import json 
import os
import sys     
import hydra
import mlflow        
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

# ── MLflow logging helpers ────────────────────────────────────────────────────
from src.ml_flow.data_prep import (
    log_dataprep_start,
    log_dataprep_params,
    log_dataprep_metrics,
    log_dataprep_lineage,
    log_dataprep_status,
)



# Import the function that does the actual work
from src.data_preparation.prepare_usecase_dataset import prepare_datasets

def count_lines(path: Path) -> int:
    """Count lines in a JSONL file = number of samples."""
    return sum(1 for _ in open(path)) if path.exists() else 0

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

    experiment_name = os.environ.get('AZUREML_ROOT_RUN_ID') 
    
    # mlflow.set_tag("mlflow.runName", f"data_prep-{config_name.replace('/', '-')}[{experiment_name}]")
    # # ── MLflow: log tags ──────────────────────────────────────
    # mlflow.set_tags({
    #     "stage":       "data_prep",
    #     "stage_enabled": cfg.pipeline.stages.data_preparation,
    #     "config_name": config_name,
    #     "config_key":  config_key,
    #     "attribute":   cfg.data_prep.attribute_col,
    #     "experiment_name": experiment_name
    # })

    # MLflow: run name + tags
    log_dataprep_start(cfg, config_name, config_key, experiment_name)

    if cfg.pipeline.stages.data_preparation:
    # if False:
        print("\n" + "="*80)
        print("STAGE 1: DATA PREPARATION")
        print("="*80)

        try:
            print("\nStarting data preparation...")
            print("\nResolved data preparation paths:")
            print(f"   CSVs       : {cfg.data_prep.input_csv_paths}")
            print(f"   Images dir : {cfg.data_prep.image_base_dir}")
            print(f"   Output dir : {cfg.data_prep.output_dir}")


            print("\nValidating inputs...")
            for csv_path in cfg.data_prep.input_csv_paths:
                if not os.path.exists(csv_path):
                    print(f"CSV not found: {csv_path}")
                    mlflow.set_tag("status", "FAILED")
                    sys.exit(1) 
                print(f"    CSV: {csv_path}")

            if not os.path.exists(cfg.data_prep.image_base_dir):
                raise RuntimeError(
                    f"Image directory not found: {cfg.data_prep.image_base_dir}"
                )
            print(f"  Image directory: {cfg.data_prep.image_base_dir}")
            
            os.makedirs(cfg.data_prep.output_dir, exist_ok=True)
            print(f"   Output: {cfg.data_prep.output_dir}")


            # # ── MLflow: log params ────────────────────────────────────
            # mlflow.log_params({
            #     "data.num_csvs":               len(cfg.data_prep.input_csv_paths),
            #     "data.index_col":              cfg.data_prep.index_col,
            #     "data.attribute_col":          cfg.data_prep.attribute_col,
            #     "data.test_size":              cfg.data_prep.test_size,
            #     "data.val_size":               cfg.data_prep.val_size,
            #     "data.random_state":           cfg.data_prep.random_state,
            #     "data.min_height_width_pixel": cfg.data_prep.min_height_width_pixel,
            #     "data.context_percent":        cfg.data_prep.context_percent,
            #     "data.output_dir":             cfg.data_prep.output_dir,
            #     "data.input_csvs":             str(OmegaConf.to_container(cfg.data_prep.input_csv_paths, resolve=True)),
            #     "data.image_base_dir":         cfg.data_prep.image_base_dir,
            #     "data.train_file":             cfg.data_prep.attribute_train_file,
            #     "data.val_file":               cfg.data_prep.attribute_val_file,
            #     "data.test_file":              cfg.data_prep.attribute_test_file,
            # })


            # ── MLflow: params ────────────────────────────────────────────────────
            log_dataprep_params(cfg)

            print("\nStarting data preparation...")
            # Run the core data preparation function
            prepare_datasets(cfg.data_prep)
            for p in Path(cfg.data_prep.output_dir).rglob("*"):
                print("   ", p)
            print("\nData preparation completed!")
            print(f"\n Output files:")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_train_file)}")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_val_file)}")
            print(f"   - {os.path.join(cfg.data_prep.output_dir, cfg.data_prep.attribute_test_file)}")
            
            print("\nData preparation completed successfully!")
            # ── MLflow: metrics + lineage ─────────────────────────────────────────
            n_train, n_val, n_test = log_dataprep_metrics(cfg)
            log_dataprep_lineage(cfg, n_train, n_val, n_test)

            # # ← ADDED: write lineage.json to output dir ────────────────────────
            # # Stored on Azure datastore alongside JSONL files — no MLflow
            # # artifact store needed.
            # # ── MLflow: log sample counts ─────────────────────────────
            # train_path = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_train_file
            # val_path   = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_val_file
            # test_path  = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_test_file

            # n_train = count_lines(train_path)
            # n_val   = count_lines(val_path)
            # n_test  = count_lines(test_path)
            # n_total = n_train + n_val + n_test

            # mlflow.log_metrics({
            #     "data.n_train_samples": n_train,
            #     "data.n_val_samples":   n_val,
            #     "data.n_test_samples":  n_test,
            #     "data.n_total_samples": n_total,
            # })
            # print(f"Split → train={n_train} | val={n_val} | test={n_test} | total={n_total}")


            # lineage = {
            #     "inputs": {
            #         "csvs":           OmegaConf.to_container(cfg.data_prep.input_csv_paths, resolve=True),
            #         "image_base_dir": cfg.data_prep.image_base_dir,
            #     },
            #     "outputs": {
            #         "train_file": cfg.data_prep.attribute_train_file,
            #         "val_file":   cfg.data_prep.attribute_val_file,
            #         "test_file":  cfg.data_prep.attribute_test_file,
            #     },
            #     "split": {
            #         "n_train": n_train,
            #         "n_val":   n_val,
            #         "n_test":  n_test,
            #         "n_total": n_total,
            #     }
            # }
            # lineage_path = Path(cfg.data_prep.output_dir) / "lineage"
            # lineage_path.mkdir(parents=True, exist_ok=True)
            # with open(lineage_path / "lineage.json", "w") as f:
            #     json.dump(lineage, f, indent=2)
            # print(f" Lineage written to: {lineage_path / 'lineage.json'}")

            # # ← ADDED: write resolved_config.yaml to output dir ───────────────
            # config_path = Path(cfg.data_prep.output_dir) / "config"
            # config_path.mkdir(parents=True, exist_ok=True)
            # with open(config_path / "resolved_config.yaml", "w") as f:
            #     f.write(OmegaConf.to_yaml(cfg))
            # print(f" Config written to: {config_path / 'resolved_config.yaml'}")

           
            # mlflow.set_tag("status", "SUCCESS")
            log_dataprep_status("SUCCESS")

        except Exception as e:
            log_dataprep_status("FAILED")
            print(f"\n❌ Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n Data preparation stage is disabled (pipeline.stages.data_preparation: false)")

if __name__ == "__main__":
    main()