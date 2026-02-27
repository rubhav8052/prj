import json
import os
from pathlib import Path

import mlflow
from omegaconf import OmegaConf

from src.ml_flow.common import (
    set_run_name,
    set_stage_tags,
    write_lineage,
    write_resolved_config,
)



def log_inference_start(cfg, config_name: str, config_key: str, experiment_name: str) -> None:
    """
    Set the run name + all standard + inference-specific tags.
    Call this once at the very beginning of the inference stage.
    """
    set_run_name("inference", config_name, experiment_name)
    set_stage_tags(
        stage           = "inference",
        stage_enabled   = cfg.pipeline.stages.inference,
        config_name     = config_name,
        config_key      = config_key,
        experiment_name = experiment_name,
        extra_tags      = {"model": cfg.get("model", "")},
    )


def log_inference_params(cfg, latest_ckpt: Path, version_dir: str, checkpoint_dir: str) -> None:
    """
    Log all inference configuration parameters to MLflow.

    Args:
        cfg:            resolved stage config
        latest_ckpt:    Path object pointing at the checkpoint directory
        version_dir:    e.g. "v4-20260201-202214"
        checkpoint_dir: e.g. "checkpoint-159"
    """
    # Pick the right model config depending on which model is active
    is_deepseek = cfg.model in ["deepseek", "all"]
    model_cfg   = cfg.inference.DEEPSEEK_CONFIG if is_deepseek else cfg.inference.QWEN_CONFIG

    mlflow.log_params({
        "inference.model":               cfg.get("model", ""),
        "inference.model_path":          str(latest_ckpt),
        "inference.version_dir":         version_dir,
        "inference.checkpoint":          checkpoint_dir,
        "inference.max_model_len":       model_cfg.get("max_model_len", ""),
        "inference.batch_size":          model_cfg.get("batch_size", ""),
        "inference.temperature":         cfg.inference.SAMPLING_PARAMS.get("temperature", ""),
        "inference.top_p":               cfg.inference.SAMPLING_PARAMS.get("top_p", ""),
        "inference.max_tokens":          cfg.inference.SAMPLING_PARAMS.get("max_tokens", ""),
        "inference.repetition_penalty":  cfg.inference.SAMPLING_PARAMS.get("repetition_penalty", ""),
    })


def log_inference_results(output_file: str, model_prefix: str) -> None:
    """
    Read a model's output JSON and log class-distribution + success metrics.

    Args:
        output_file:  path to the inference output JSON (list of result dicts)
        model_prefix: MLflow metric prefix, e.g. "inference.deepseek"
    """
    if not os.path.exists(output_file):
        print(f"Output file not found for metrics: {output_file}")
        return

    with open(output_file) as f:
        results = json.load(f)

    n_total   = len(results)
    n_failed  = sum(1 for r in results if not r.get("output") or r.get("output") == "")
    n_success = n_total - n_failed
    success_rate = round((n_success / n_total) * 100, 2) if n_total > 0 else 0

    # Build class distribution
    class_counts: dict[str, int] = {}
    for r in results:
        label = r.get("output", "").strip()
        if label:
            class_counts[label] = class_counts.get(label, 0) + 1

    # Summary metrics
    mlflow.log_metrics({
        "n_total_images": n_total,
        "n_success":      n_success,
        "n_failed":       n_failed,
        "success_rate":   success_rate,
    })

    # Per-class count + percentage
    for cls, count in class_counts.items():
        safe_cls = cls.replace(" ", "_")
        mlflow.log_metrics({
            f"{safe_cls}.count":   count,
            f"{safe_cls}.percent": round((count / n_total) * 100, 2),
        })

    print(f"{model_prefix} results: total={n_total} | success={n_success} | failed={n_failed}")
    print(f"Class distribution: {class_counts}")


def log_inference_lineage(cfg, latest_ckpt: Path, version_dir: str, checkpoint_dir: str) -> None:
    """
    Write lineage.json and resolved_config.yaml to the inference output directory.
    """
    output_file = (
        cfg.inference.DEEPSEEK_CONFIG.output_file
        if cfg.model in ["deepseek", "all"]
        else cfg.inference.QWEN_CONFIG.output_file
    )

    lineage = {
        "inputs": {
            "model_path":   str(latest_ckpt),
            "version_dir":  version_dir,
            "checkpoint":   checkpoint_dir,
            "image_folder": str(cfg.inference.paths.image_folder),
        },
        "outputs": {
            "inference_output_dir": str(cfg.inference.paths.inference_output_dir),
            "output_file":          output_file,
        },
        "config": {
            "model":               cfg.get("model", ""),
            "temperature":         cfg.inference.SAMPLING_PARAMS.get("temperature", ""),
            "top_p":               cfg.inference.SAMPLING_PARAMS.get("top_p", ""),
            "max_tokens":          cfg.inference.SAMPLING_PARAMS.get("max_tokens", ""),
            "repetition_penalty":  cfg.inference.SAMPLING_PARAMS.get("repetition_penalty", ""),
        },
    }
    write_lineage(cfg.inference.paths.inference_output_dir, lineage)
    write_resolved_config(cfg.inference.paths.inference_output_dir, cfg)


def log_inference_status(status: str) -> None:
    """
    Set the final status tag.  status should be "SUCCESS" or "FAILED".
    """
    mlflow.set_tag("status", status)
