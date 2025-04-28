# VLLM Batch Inference

This document describes how to run batch inference using the VLLM library with pre-trained or fine-tuned DeepSeek-VL and Qwen-VL models.

## Overview

This inference pipeline uses the `vllm` library for optimized inference. It provides a base class and specific implementations for different VLMs.

*   **Core Library:** `vllm`
*   **Base Class:** `src/inference/local_lvms/base_inference.py` (Defines the common structure)
    
*   **Implementations:**
    *   `src/inference/local_lvms/deepseek_inference.py`
        
    *   `src/inference/local_lvms/qwen_inference.py`
        
*   **Entry Script:** `src/scripts/run_inference.py`

The pipeline loads images from a folder, prepares batches according to the specific model's requirements, runs inference using the VLLM engine, and saves the results to a JSON file.

## Configuration

VLLM inference configuration is primarily managed within `src/config/inference_config.py` and command-line arguments passed to `src/scripts/run_inference.py`.

*   **Main Config File:** `src/config/inference_config.py`

    *   Defines model-specific dictionaries (`DEEPSEEK_CONFIG`, `QWEN_CONFIG`) containing parameters like `model_name`, `max_model_len`, `batch_size`, `output_file`, and default prompt locations.
    *   Defines common `SAMPLING_PARAMS` used by VLLM.

*   **Command-Line Arguments (`src/scripts/run_inference.py`):**
    
    *   `--model`: Selects the model to run (`deepseek`, `qwen`, or `all`).
    *   `--image_folder`: Overrides the default image input folder.
    *   `--output_dir`: Specifies the directory to save output JSON files (overrides the directory part of `output_file` in the config).
    *   `--prompt_module`, `--prompt_variable`: Allows overriding the default prompt by specifying a Python module and variable name containing the prompt string.

*   **Modifying Configuration:**
    1.  **Model Parameters:** Modify the `DEEPSEEK_CONFIG` or `QWEN_CONFIG` dictionaries in `src/config/inference_config.py` for persistent changes to model paths, batch sizes, etc.
    2.  **Sampling Parameters:** Adjust the `SAMPLING_PARAMS` dictionary in `src/config/inference_config.py`.
    3.  **Runtime Overrides:** Use the command-line arguments (`--image_folder`, `--output_dir`, `--prompt_module`, `--prompt_variable`) when running `src/scripts/run_inference.py` for temporary or run-specific changes.

## Local Execution

Run the inference script from the root directory:

```bash
# Run Deepseek model with default settings
python src/scripts/run_inference.py --model deepseek
# Run Qwen model with custom image folder and output directory
python src/scripts/run_inference.py --model qwen --image_folder /path/to/my/images --output_dir /path/to/my/results
# Run both models with a custom prompt
python src/scripts/run_inference.py --model all --prompt_module src.prompts.my_custom_prompts --prompt_variable my_specific_prompt
```


The script `src/scripts/run_inference.py` (Lines: 50-96) parses arguments, loads the appropriate configuration from `src/config/inference_config.py`, potentially overrides the prompt, instantiates the correct inference class (`DeepseekInference` or `QwenInference`), and calls the `batch_inference` method.

## Azure ML Execution

The design document (`docs/design/vlm_batch_inference.md`, Lines: 112-131) indicates using `azure_submit.py` with specific run configs for VLLM inference on Azure ML.

1.  **Configure AML Run Config:** Use or adapt `.yml` files like `deployment/run_configs/deepseek_inference.yml` or `deployment/run_configs/qwen_inference.yml`. **Note:** These file names might be the same as those potentially used for Swift inference; ensure the `ENTRY_SCRIPT` and `ENV` are set correctly for VLLM. The VLLM environment might differ from the Swift one (e.g., different base Docker image or dependencies).
    ```yaml
    # Example: deployment/run_configs/vllm_deepseek_inference.yml
    ENV: "vllm_environment_name" # Ensure this env has VLLM installed
    EXPERIMENT: "vllm_inference"
    COMPUTE: "gpu_compute_target" # e.g., gpupoola100
    RUN_NAME: "vllm_deepseek_run"
    NODES: 1
    SCRIPT_ARGUMENTS:
      model: deepseek # Corresponds to --model argument
      image_folder: /azureml/input/my_images # Example path in AML context
      output_dir: /azureml/output/results # Example path in AML context
      # prompt_module: src.prompts.custom # Optional
      # prompt_variable: my_prompt # Optional
    DEBUG: False
    SRC: "."
    ENTRY_SCRIPT: "src/scripts/run_inference.py" # VLLM entry script
    ```
    *   Set `ENV` to your VLLM-enabled Azure ML Environment.
    *   Set `EXPERIMENT`, `COMPUTE`, `RUN_NAME`.
    *   Under `SCRIPT_ARGUMENTS`, provide the arguments for `src/scripts/run_inference.py` (e.g., `model`, `image_folder`, `output_dir`). Note that keys are lowercase. Input/output paths should typically reference Azure ML paths (e.g., mounted datastores, output folders).
    *   Ensure `ENTRY_SCRIPT` is `src/scripts/run_inference.py`.

2.  **Submit the Job:**
    ```bash
    # Example using a hypothetical VLLM-specific config
    python azure_submit.py deployment/run_configs/vllm_deepseek_inference.yml
    ```

This executes `src/scripts/run_inference.py` on Azure ML, using the arguments provided in the `.yml` file.