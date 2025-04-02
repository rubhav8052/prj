# VLM Batch Inference Design

This document outlines the design and architecture of the VLM (Vision-Language Model) batch inference system.

## Overview

The VLM batch inference system provides a structured framework for running batch inference with various Vision-Language Models on vehicle images. It's designed to be modular, extensible, and efficient, allowing for easy addition of new models and customization of inference parameters.

### Key Components

1. **Configuration (`src/config/inference_config.py`)**
   - Centralized configuration for all models
   - Defines model parameters, batch sizes, and sampling parameters
   - Configurable output paths

2. **Base Inference (`src/inference/local_lvms/base_inference.py`)**
   - Abstract base class defining the inference interface
   - Implements common functionality:
     - Batch processing
     - File I/O
     - Performance metrics
     - Progress tracking

3. **Model-Specific Implementations**
   - `src/inference/local_lvms/deepseek_inference.py`
   - `src/inference/local_lvms/qwen_inference.py`
   - Each implements model-specific initialization and batch preparation

4. **Run Inference Script (`src/scripts/run_inference.py`)**
   - Command-line interface
   - Model selection
   - Output directory configuration

## Data Flow

1. User invokes the run_inference script with desired parameters
2. Script initializes the selected model(s) with appropriate configuration
3. For each model:
   - Images are loaded and processed in batches
   - Model generates predictions for each batch
   - Results are written to JSON output file
   - Performance metrics are calculated and displayed


## Extension Points

### Adding New Models

To add a new VLM model:

1. Create a new implementation in `src/inference/local_lvms/` that inherits from `BaseInference`
2. Implement the required methods:
   - `_initialize_llm()`: Initialize the model
   - `prepare_batch()`: Prepare data for batch inference
3. Add model configuration to `src/config/inference_config.py`
4. Update `src/scripts/run_inference.py` to include the new model option

### Customizing Output Format

The output format can be customized by modifying the `batch_inference` method in the `BaseInference` class.

## How to Run

### Prerequisites

1. Ensure all dependencies are installed:

2. Make sure the model weights are accessible:
   - Deepseek model: Available on Hugging Face (`deepseek-ai/deepseek-vl2-tiny`)
   - Qwen model: Available on Hugging Face (`Qwen/Qwen2-VL-2B-Instruct`)

### Running Inference Locally

#### Basic Usage

Run inference with default settings:

```bash
python -m src.scripts.run_inference
```

This will process all images in the default folder using both Deepseek and Qwen models.

#### Command-line Options

```bash
python -m src.scripts.run_inference --model [deepseek|qwen|all] --image_folder /path/to/images --output_dir /path/to/output
```

Arguments:
- `--model`: Model to use for inference (choices: "deepseek", "qwen", "all"; default: "all")
- `--image_folder`: Path to folder containing images (default: configured in inference_config.py)
- `--output_dir`: Directory to save results (default: "outputs")

#### Examples

Run only the Deepseek model:
```bash
python -m src.scripts.run_inference --model deepseek
```

Run the Qwen model with a custom image folder:
```bash
python -m src.scripts.run_inference --model qwen --image_folder /data/vehicle_images
```

Run both models with a custom output directory:
```bash
python -m src.scripts.run_inference --model all --output_dir results/vehicle_analysis
```

### Running on Azure Machine Learning

For running inference at scale on Azure Machine Learning, use the provided configuration files:

```bash
python azure_submit.py deployment/run_configs/qwen_inference.yml
```

This will:
1. Create an AML job with the specified configuration
2. Submit the job to the AML workspace
3. Use the compute resources defined in the configuration file
4. Run the inference pipeline on the specified data
5. Store the results in the configured output location

You can also run the Deepseek model on AML:

```bash
python azure_submit.py deployment/run_configs/deepseek_inference.yml
```

### Output

Results are saved as JSON files in the specified output directory. Each file contains an array of objects with the following structure:

```json
[
    {
        "image_name": "vehicle_001.jpg",
        "output": "JSON output from the model..."
    },
    {
        "image_name": "vehicle_002.jpg",
        "output": "JSON output from the model..."
    }
]
```

The system also displays performance metrics in the console:
- Total inference time (excluding warmup)
- Average inference time per batch
- Average inference time per image
```
