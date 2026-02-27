import json
import os
import sys
from pathlib import Path

import mlflow
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig


def set_run_name(stage: str, config_name: str, experiment_name: str) -> None:
    """
    Set the MLflow run display name.

    Pattern: "<stage>-<config_name_with_dashes>[<experiment_name>]"
    Example: "data_prep-config1-default[<run_id>]"
    """
    run_name = f"{stage}-{config_name.replace('/', '-')}[{experiment_name}]"
    mlflow.set_tag("mlflow.runName", run_name)


def set_stage_tags(
    stage: str,
    stage_enabled: bool,
    config_name: str,
    config_key: str,
    experiment_name: str,
    extra_tags: dict | None = None,
) -> None:

    tags = {
        "stage":           stage,
        "stage_enabled":   stage_enabled,
        "config_name":     config_name,
        "config_key":      config_key,
        "experiment_name": experiment_name,
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)




def write_lineage(output_dir: str | Path, lineage: dict) -> Path:

    lineage_dir = Path(output_dir) / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    lineage_file = lineage_dir / "lineage.json"
    with open(lineage_file, "w") as f:
        json.dump(lineage, f, indent=2)
    print(f"Lineage written to: {lineage_file}")
    return lineage_file


def write_resolved_config(output_dir: str | Path, cfg: DictConfig) -> Path:

    config_dir = Path(output_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "resolved_config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Config written to: {config_file}")
    return config_file



def count_lines(path: str | Path) -> int:
    """
    Count lines in a JSONL file (= number of samples).
    Returns 0 if the file does not exist.
    """
    p = Path(path)
    return sum(1 for _ in open(p)) if p.exists() else 0


