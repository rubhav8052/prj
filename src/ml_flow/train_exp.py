import json
from pathlib import Path

import mlflow
from omegaconf import OmegaConf, ListConfig

from src.ml_flow.common import (
    set_run_name,
    set_stage_tags,
    write_lineage,
    write_resolved_config,
    count_lines,
)


def log_training_start(cfg, config_name: str, config_key: str, experiment_name: str) -> None:
    """
    Set the run name + all standard + training-specific tags.
    Call this once at the very beginning of the training stage.
    """
    set_run_name("train_export", config_name, experiment_name)
    set_stage_tags(
        stage           = "train_export",
        stage_enabled   = cfg.pipeline.stages.training,
        config_name     = config_name,
        config_key      = config_key,
        experiment_name = experiment_name,
        extra_tags      = {"model_id": cfg.training.model_id},
    )



def log_training_params(cfg) -> None:
    """
    Log all training hyper-parameters / config values to MLflow.
    Also logs derived metrics: effective_batch_size, n_train_samples, n_val_samples.
    """
    train_dataset_paths = cfg.training.get('train_dataset_paths', [])
    val_dataset_paths   = cfg.training.get('val_dataset_paths', [])

    if isinstance(train_dataset_paths, str):
        train_dataset_paths = [train_dataset_paths]
    if isinstance(val_dataset_paths, str):
        val_dataset_paths = [val_dataset_paths]

    mlflow.log_params({
        "train.model_id":                    cfg.training.model_id,
        "train.train_type":                  cfg.training.get("train_type", ""),
        "train.lora_rank":                   cfg.training.get("lora_rank", ""),
        "train.freeze_vit":                  cfg.training.get("freeze_vit", ""),
        "train.freeze_parameters_ratio":     cfg.training.get("freeze_parameters_ratio", ""),
        "train.num_train_epochs":            cfg.training.get("num_train_epochs", ""),
        "train.batch_size":                  cfg.training.get("batch_size", ""),
        "train.gradient_accumulation_steps": cfg.training.get("gradient_accumulation_steps", ""),
        "train.learning_rate":               cfg.training.get("learning_rate", ""),
        "train.loss_type":                   cfg.training.get("loss_type", ""),
        "train.torch_dtype":                 cfg.training.get("torch_dtype", ""),
        "train.output_dir":                  cfg.training.output_dir,
    })

    # Derived metrics
    train_path = train_dataset_paths[0] if isinstance(train_dataset_paths, (list, ListConfig)) else train_dataset_paths
    val_path   = val_dataset_paths[0]   if isinstance(val_dataset_paths,   (list, ListConfig)) else val_dataset_paths

    n_train        = count_lines(train_path)
    n_val          = count_lines(val_path)
    effective_batch = (
        cfg.training.get("batch_size", 1) * cfg.training.get("gradient_accumulation_steps", 1)
    )
    mlflow.log_metrics({
        "train.effective_batch_size": effective_batch,
        "train.n_train_samples":      n_train,
        "train.n_val_samples":        n_val,
    })




def log_training_checkpoints(cfg) -> None:
    """
    After training completes, scan the output directory for checkpoints,
    read trainer_state.json, and log all summary + curve metrics.
    """
    checkpoints = sorted(
        [d for d in Path(cfg.training.output_dir).rglob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[1])
    )

    if not checkpoints:
        print("No checkpoints found — skipping checkpoint metrics.")
        return

    latest_ckpt      = checkpoints[-1]
    latest_ckpt_step = int(latest_ckpt.name.split("-")[1])

    mlflow.log_metric("train.total_checkpoints_saved", len(checkpoints))
    mlflow.log_metric("train.final_checkpoint_step",   latest_ckpt_step)
    mlflow.set_tag("train.latest_checkpoint", str(latest_ckpt))

    trainer_state_path = latest_ckpt / "trainer_state.json"
    if not trainer_state_path.exists():
        print(f"trainer_state.json not found at: {trainer_state_path}")
        return

    with open(trainer_state_path) as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])
    best_ckpt   = trainer_state.get("best_model_checkpoint", "")

    # Summary metrics
    mlflow.log_metrics({
        "train.best_loss":          trainer_state.get("best_metric", 0),
        "train.final_epoch":        trainer_state.get("epoch", 0),
        "train.actual_total_steps": trainer_state.get("global_step", 0),
        "train.max_steps":          trainer_state.get("max_steps", 0),
        "train.train_batch_size":   trainer_state.get("train_batch_size", 0),
    })
    mlflow.set_tag("train.best_checkpoint", best_ckpt)
    mlflow.set_tag("train.best_is_final",   str(best_ckpt == str(latest_ckpt)))

    # Loss improvement
    train_entries = [e for e in log_history if "loss" in e]
    if len(train_entries) >= 2:
        first_loss           = train_entries[0]["loss"]
        last_loss            = train_entries[-1]["loss"]
        loss_improvement     = round(first_loss - last_loss, 4)
        loss_improvement_pct = round((loss_improvement / first_loss) * 100, 2) if first_loss > 0 else 0
        mlflow.log_metrics({
            "train.first_loss":           first_loss,
            "train.last_loss":            last_loss,
            "train.loss_improvement":     loss_improvement,
            "train.loss_improvement_pct": loss_improvement_pct,
        })

    # Eval loss improvement
    eval_entries = [e for e in log_history if "eval_loss" in e]
    if eval_entries:
        mlflow.log_metrics({
            "train.first_eval_loss": eval_entries[0]["eval_loss"],
            "train.last_eval_loss":  eval_entries[-1]["eval_loss"],
            "train.best_eval_loss":  min(e["eval_loss"] for e in eval_entries),
        })

    # Token accuracy
    token_entries      = [e for e in log_history if "token_acc" in e]
    eval_token_entries = [e for e in log_history if "eval_token_acc" in e]
    if token_entries:
        mlflow.log_metrics({
            "train.first_token_acc": token_entries[0]["token_acc"],
            "train.last_token_acc":  token_entries[-1]["token_acc"],
            "train.best_eval_token_acc": max(
                e["eval_token_acc"] for e in eval_token_entries
            ) if eval_token_entries else 0,
        })

    # Final training speed + memory
    if train_entries:
        last_entry = train_entries[-1]
        mlflow.log_metrics({
            "train.final_train_speed_iter_per_s": last_entry.get("train_speed(iter/s)", 0),
            "train.final_memory_gb":              last_entry.get("memory(GiB)", 0),
        })

    # Per-epoch eval metrics (gives epoch-level graphs in MLflow UI)
    for entry in log_history:
        if "eval_loss" in entry:
            epoch = round(entry.get("epoch", 0))
            mlflow.log_metric("train.eval_loss_per_epoch",      entry["eval_loss"],            step=epoch)
            mlflow.log_metric("train.eval_token_acc_per_epoch", entry.get("eval_token_acc", 0), step=epoch)

    print("Logged summary + curve metrics from trainer_state.json")


def log_export_params(export_cfg, version_dir: str, checkpoint_name: str) -> None:
    """
    Log export/quantization parameters and tags to MLflow.
    """
    mlflow.log_params({
        "export.model":       export_cfg.get("model", ""),
        "export.lora_path":   str(export_cfg.lora_paths[0]),
        "export.output_dir":  export_cfg.output_dir,
        "export.merge_lora":  export_cfg.get("merge_lora", True),
    })
    mlflow.set_tags({
        "export.version_dir": version_dir,
        "export.checkpoint":  checkpoint_name,
    })
    mlflow.log_metric("export.checkpoint_step", int(checkpoint_name.split("-")[1]))

def log_export_metrics(export_cfg) -> None:
    """
    Log the size on disk of the exported model.
    """
    export_path = Path(export_cfg.output_dir)
    if export_path.exists():
        export_size_gb = sum(
            f.stat().st_size for f in export_path.rglob("*") if f.is_file()
        ) / (1024 ** 3)
        mlflow.log_metric("export.model_size_gb", round(export_size_gb, 3))
        print(f"Exported model size: {export_size_gb:.3f} GB")

def log_training_lineage(cfg, export_cfg, version_dir: str, checkpoint_dir: str) -> None:
    """
    Write lineage.json and resolved_config.yaml for the export output directory.
    """
    lineage = {
        "inputs": {
            "model_id":       cfg.training.model_id,
            "lora_checkpoint": str(export_cfg.lora_paths[0]),
            "train_datasets": OmegaConf.to_container(
                cfg.training.get('train_dataset_paths', []), resolve=True
            ),
        },
        "outputs": {
            "export_dir": str(export_cfg.output_dir),
        },
    }
    write_lineage(export_cfg.output_dir, lineage)
    write_resolved_config(export_cfg.output_dir, cfg)


def log_training_status(scope: str, status: str) -> None:
    """
    Set a scoped status tag.
    scope:  "train" | "export"
    status: "SUCCESS" | "FAILED"

    Results in tags like  train.status = SUCCESS
    """
    mlflow.set_tag(f"{scope}.status", status)
