# Swift VLM Workflow: Training, Inference, and Export

This document covers the workflow using the Swift framework for fine-tuning, running inference, and exporting Vision-Language Models (VLMs).

## Overview

This workflow utilizes the `ms-swift` library for:

1.  **Supervised Fine-Tuning (SFT):** Fine-tuning a base VLM (like DeepSeek-VL or Qwen-VL) on a custom dataset using techniques like LoRA, potentially freezing parts of the Vision Transformer (ViT) or the Language Model (LLM).
2.  **Inference:** Running the fine-tuned (or base) model on a dataset to generate predictions using the Swift `PtEngine`.
3.  **Export:** Merging LoRA adapters into the base model weights for deployment.

## 1. Swift Training (SFT)

### Configuration

Training is configured using a Python file containing an instance of the `TrainingConfig` dataclass.

*   **Base Configuration Class:** `src/config/training_config.py`
*   **Specific Configuration Files:** Examples like `src/config/training_configs/qwen2b-occlusion.py` or `src/config/training_configs/deepseek-vehicle_type.py` show how to set parameters for specific experiments.
*   **Key Parameters:**
    *   `model_id`: Identifier of the base model (Hugging Face or ModelScope).
    *   `train_dataset_paths`, `val_dataset_paths`: Paths to the training and validation JSONL datasets (prepared using the Data Prep pipeline).
    *   `output_dir`: Directory to save checkpoints and logs.
    *   `train_type`: Training method (e.g., 'lora').
    *   `lora_rank`: Rank for LoRA adapters.
    *   `freeze_vit`: Whether to freeze the Vision Transformer weights.
    *   `freeze_parameters_ratio`: Ratio of LLM parameters to freeze (0.0 trains all, 0.9 freezes 90%).
    *   `num_train_epochs`, `batch_size`, `learning_rate`, `max_length`: Standard training hyperparameters.
    *   `torch_dtype`: Data type for training (e.g., 'bfloat16').
    *   `swift_libs_path`: Path to potentially isolated Swift/transformer libraries (e.g., `/deepseek_libs`, `/qwen_libs`).

*   **Modifying Configuration:**
    1.  **Recommended:** Create a new Python configuration file in `src/config/training_configs/` (e.g., `my_experiment_config.py`).
    2.  Import `TrainingConfig` from `src.config.training_config`.
    3.  Create an instance: `config = TrainingConfig()`.
    4.  Override parameters as needed: `config.model_id = "new/model"`, `config.num_train_epochs = 5`, etc.
    5.  Pass the path to this file via the `--config` argument.

### Local Execution

Run the training script from the root directory:

```bash
python src/scripts/run_training.py --config [PATH_TO_TRAINING_CONFIG_PY]
```


*   Replace `[PATH_TO_TRAINING_CONFIG_PY]` with the path to your configuration file (e.g., `src/config/training_configs/qwen2b-occlusion.py`).

The script `src/scripts/run_training.py` (Lines: 1-42) loads the config, sets up the environment (including `swift_libs_path`), and calls `run_swift_sft` from `src/training/swift_trainer.py` (Lines: 6-74), which prepares arguments and executes Swift's `sft_main`.

### Azure ML Execution

1.  **Create/Configure AML Run Config:** Define a `.yml` file (e.g., `deployment/run_configs/training.yml` - you might need to create this based on others like `qwen_inference.yml`).
    ```yaml
    # Example: deployment/run_configs/my_training_job.yml
    ENV: "swift" # Or your custom training environment
    EXPERIMENT: "swift_training"
    COMPUTE: "gpu_compute_target" # e.g., gpupoola100
    RUN_NAME: "my_model_training_run"
    NODES: 1 # Or more for distributed training if configured
    SCRIPT_ARGUMENTS:
      CONFIG: src/config/training_configs/my_experiment_config.py # Path to your specific config
    DEBUG: False
    SRC: "."
    ENTRY_SCRIPT: "src/scripts/run_training.py"
    ```
2.  **Submit the Job:**
    ```bash
    python azure_submit.py deployment/run_configs/my_training_job.yml
    ```

## 2. Swift Inference

### Configuration

Inference using the Swift engine is configured via a Python file with a `SwiftInferenceConfig` instance.

*   **Base Configuration Class:** `src/config/swift_inference_config.py`
*   **Specific Configuration Files:** Examples like `src/config/inference_configs/qwen2b-occlusion.py` or `src/config/inference_configs/deepseek-vehicle_type.py` define specific inference tasks.
*   **Key Parameters:**
    *   `model_id`: Base model identifier.
    *   `lora_path`: Path to the LoRA checkpoint directory (e.g., from training output). The script will try to find the latest checkpoint within this directory.
    *   `dataset_path`: Path to the input JSONL dataset for inference.
    *   `output_dir`: Directory to save the inference results.
    *   `output_filename_prefix`: Prefix for the output JSONL file.
    *   `batch_size`: Inference batch size.
    *   `dtype`: Data type for inference ('bfloat16', 'float16').
    *   `request_config`: An instance of `SwiftRequestConfig` defining generation parameters (max_tokens, temperature, etc.).
    *   `swift_libs_path`: Path to potentially isolated Swift/transformer libraries.

*   **Modifying Configuration:**
    1.  **Recommended:** Create a new Python configuration file in `src/config/inference_configs/` (e.g., `my_inference_task.py`).
    2.  Import `SwiftInferenceConfig` from `src.config.swift_inference_config`.
    3.  Create an instance: `config = SwiftInferenceConfig()`.
    4.  Override parameters: `config.lora_path = "/path/to/my/lora_checkpoint"`, `config.dataset_path = "/path/to/my/test_data.jsonl"`, etc.
    5.  Pass the path to this file via the `--config` argument.

### Local Execution

Run the Swift inference script from the root directory:

```bash
python src/scripts/run_swift_inference.py --config [PATH_TO_INFERENCE_CONFIG_PY]
```


*   Replace `[PATH_TO_INFERENCE_CONFIG_PY]` with the path to your configuration file (e.g., `src/config/inference_configs/qwen2b-occlusion.py`).

The script `src/scripts/run_swift_inference.py` (Lines: 1-47) loads the config, sets up the environment (including `swift_libs_path`), and calls `run_swift_inference` from `src/inference/local_lvms/swift_inference/engine.py` (Lines: 17-197), which handles model loading (with LoRA), engine initialization, and batch inference.

### Azure ML Execution

1.  **Configure AML Run Config:** Use or adapt existing `.yml` files like `deployment/run_configs/qwen_inference.yml` or `deployment/run_configs/deepseek_inference.yml` (Note: the design doc mentions these for VLLM, ensure you have Swift-specific ones if needed, or adapt these).
    *   Set `ENV` to your Swift-enabled environment.
    *   Set `EXPERIMENT`, `COMPUTE`, `RUN_NAME`.
    *   Under `SCRIPT_ARGUMENTS`, set `CONFIG` to the path of your specific Python inference configuration file (e.g., `src/config/inference_configs/qwen2b-occlusion.py`).
    *   Ensure `ENTRY_SCRIPT` is `src/scripts/run_swift_inference.py`.

2.  **Submit the Job:**
    ```bash
    # Example using the Qwen config file
    python azure_submit.py deployment/run_configs/qwen_inference.yml
    ```

## 3. Swift Model Export (LoRA Merge)

### Configuration

Model export (specifically merging LoRA adapters) is configured using a Python file with an `ExportConfig` instance.

*   **Base Configuration Class:** `src/config/export_config.py`
*   **Key Parameters:**
    *   `model`: Base model identifier.
    *   `lora_paths`: *List* of paths to LoRA checkpoint directories to merge.
    *   `output_dir`: Directory to save the merged model.
    *   `merge_lora`: Boolean flag, should be `True` to perform the merge.
    *   `dataset_path`: Path to a dataset (can be small, sometimes needed for model loading).
    *   `swift_libs_path`: Path to potentially isolated Swift/transformer libraries.
    *   *(Note: Quantization parameters like `quant_bits`, `quant_method` are commented out in the provided `ExportConfig` and `run_swift_export` but could be added back if needed)*.

*   **Modifying Configuration:**
    1.  **Recommended:** Create a new Python configuration file in `src/config/export_configs/` (e.g., `my_export_job.py`).
    2.  Import `ExportConfig` from `src.config.export_config`.
    3.  Create an instance: `config = ExportConfig()`.
    4.  Override parameters: `config.model = "base/model/to/merge/into"`, `config.lora_paths = ["/path/to/lora1", "/path/to/lora2"]`, `config.output_dir = "/path/to/merged_model"`, etc.
    5.  Pass the path to this file via the `--config` argument.

### Local Execution

Run the export script from the root directory:

```bash
python src/scripts/run_export.py --config [PATH_TO_EXPORT_CONFIG_PY]
```


*   Replace `[PATH_TO_EXPORT_CONFIG_PY]` with the path to your configuration file.

The script `src/scripts/run_export.py` (Lines: 1-43) loads the config, sets up the environment, and calls `run_swift_export` from `src/inference/local_lvms/swift_inference/exporter.py` (Lines: 10-46), which uses Swift's `export_main` with `ExportArguments`.

### Azure ML Execution

1.  **Configure AML Run Config:** Use or adapt `deployment/run_configs/export.yml`.
    *   Set `ENV`, `EXPERIMENT`, `COMPUTE`, `RUN_NAME`.
    *   Under `SCRIPT_ARGUMENTS`, set `CONFIG` to the path of your specific Python export configuration file.
    *   Ensure `ENTRY_SCRIPT` is `src/scripts/run_export.py`.

2.  **Submit the Job:**
    ```bash
    python azure_submit.py deployment/run_configs/export.yml
    ```