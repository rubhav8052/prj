
import mlflow
from omegaconf import OmegaConf
from pathlib import Path

from src.ml_flow.common import (
    set_run_name,
    set_stage_tags,
    write_lineage,
    write_resolved_config,
    count_lines,
)




def log_dataprep_start(cfg, config_name: str, config_key: str, experiment_name: str) -> None:
    """
    Set the run name + all standard + data-prep-specific tags.
    Call this once at the very beginning of the data-prep stage.
    """
    set_run_name("data_prep", config_name, experiment_name)
    set_stage_tags(
        stage           = "data_prep",
        stage_enabled   = cfg.pipeline.stages.data_preparation,
        config_name     = config_name,
        config_key      = config_key,
        experiment_name = experiment_name,
        extra_tags      = {"attribute": cfg.data_prep.attribute_col},
    )



def log_dataprep_params(cfg) -> None:
    """
    Log all data-preparation hyper-parameters / config values to MLflow.
    """
    mlflow.log_params({
        "data.num_csvs":               len(cfg.data_prep.input_csv_paths),
        "data.index_col":              cfg.data_prep.index_col,
        "data.attribute_col":          cfg.data_prep.attribute_col,
        "data.test_size":              cfg.data_prep.test_size,
        "data.val_size":               cfg.data_prep.val_size,
        "data.random_state":           cfg.data_prep.random_state,
        "data.min_height_width_pixel": cfg.data_prep.min_height_width_pixel,
        "data.context_percent":        cfg.data_prep.context_percent,
        "data.output_dir":             cfg.data_prep.output_dir,
        "data.input_csvs":             str(OmegaConf.to_container(cfg.data_prep.input_csv_paths, resolve=True)),
        "data.image_base_dir":         cfg.data_prep.image_base_dir,
        "data.train_file":             cfg.data_prep.attribute_train_file,
        "data.val_file":               cfg.data_prep.attribute_val_file,
        "data.test_file":              cfg.data_prep.attribute_test_file,
    })



def log_dataprep_metrics(cfg) -> tuple[int, int, int]:
    """
    Count lines in the output JSONL files and log sample-count metrics.

    Returns:
        (n_train, n_val, n_test)
    """
    train_path = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_train_file
    val_path   = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_val_file
    test_path  = Path(cfg.data_prep.output_dir) / cfg.data_prep.attribute_test_file

    n_train = count_lines(train_path)
    n_val   = count_lines(val_path)
    n_test  = count_lines(test_path)
    n_total = n_train + n_val + n_test

    mlflow.log_metrics({
        "data.n_train_samples": n_train,
        "data.n_val_samples":   n_val,
        "data.n_test_samples":  n_test,
        "data.n_total_samples": n_total,
    })
    print(f"Split → train={n_train} | val={n_val} | test={n_test} | total={n_total}")
    return n_train, n_val, n_test


def log_dataprep_lineage(cfg, n_train: int, n_val: int, n_test: int) -> None:
    """
    Write lineage.json and resolved_config.yaml to the output directory.
    """
    lineage = {
        "inputs": {
            "csvs":           OmegaConf.to_container(cfg.data_prep.input_csv_paths, resolve=True),
            "image_base_dir": cfg.data_prep.image_base_dir,
        },
        "outputs": {
            "train_file": cfg.data_prep.attribute_train_file,
            "val_file":   cfg.data_prep.attribute_val_file,
            "test_file":  cfg.data_prep.attribute_test_file,
        },
        "split": {
            "n_train": n_train,
            "n_val":   n_val,
            "n_test":  n_test,
            "n_total": n_train + n_val + n_test,
        },
    }
    write_lineage(cfg.data_prep.output_dir, lineage)
    write_resolved_config(cfg.data_prep.output_dir, cfg)

def log_dataprep_status(status: str) -> None:
    """
    Set the final status tag.  status should be "SUCCESS" or "FAILED".
    """
    mlflow.set_tag("status", status)