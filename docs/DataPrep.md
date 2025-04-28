# Data Preparation Pipeline

This document describes the process for preparing vehicle datasets from CSV files into the JSONL format required for VLM training and inference.

## Overview

The pipeline performs the following steps:

1.  Loads vehicle data from one or more input CSV files.
2.  Extracts bounding box information and relevant attributes (e.g., occlusion, vehicle type).
3.  Processes images:
    *   Resolves image paths based on a base directory.
    *   Crops the image around the bounding box, potentially adding context.
    *   Saves the processed images to an output directory.
4.  Generates prompts based on a template file and the extracted attributes.
5.  Formats the data (image paths, prompts, ground truth responses) into JSONL files.
6.  Splits the data into training and validation sets.

## Configuration

The data preparation process is configured using a Python file containing an instance of the `DataPrepConfig` dataclass.

*   **Base Configuration Class:** `src/config/data_prep_config.py`
    ```python:src/config/data_prep_config.py
    startLine: 5
    endLine: 35
    ```
*   **Key Parameters:**
    *   `input_csv_paths`: List of paths to the input CSV files.
    *   `image_base_dir`: Base directory to find the original images referenced in the CSVs.
    *   `output_dir`: Directory to save the processed images (in an `images` subfolder) and the output JSONL files.
    *   `attribute_col`: Name of the column in the CSV containing the target attribute (e.g., "occlusion", "type").
    *   `attribute_prompt_file`: Path to a markdown file containing the prompt template.
    *   `context_percent`: Percentage of context to add around the bounding box when cropping.
    *   `test_size`: Fraction of the data to use for the validation set.
    *   `num_threads`: Number of parallel threads for image processing.
    *   `img_sha_col`, `bbox_cols`, `index_col`: Column names for image identifiers, bounding boxes, and unique indices.
    *   `attribute_train_file`, `attribute_val_file`: Output filenames for the JSONL files.

*   **Modifying Configuration:**
    1.  You can directly modify the default values in `src/config/data_prep_config.py`.
    2.  **Recommended:** Create a new Python configuration file (e.g., `src/config/data_prep_configs/my_custom_prep.py`) that imports `DataPrepConfig`, creates an instance, and overrides the desired parameters:
        ```python
        # src/config/data_prep_configs/my_custom_prep.py
        import os
        from src.config.data_prep_config import DataPrepConfig

        config = DataPrepConfig()
        config.input_csv_paths = ["/path/to/your/data.csv"]
        config.output_dir = "/path/to/your/output"
        config.attribute_col = "vehicle_color"
        config.attribute_prompt_file = "/path/to/color_prompt.md"
        # ... override other parameters as needed
        ```
    3.  Pass the path to your custom config file using the `--config` argument when running the script locally or via the AML `.yml` file.

## Local Execution

Run the preparation script from the root directory of the project:

```bash
python src/scripts/prepare_dataset.py --config [PATH_TO_CONFIG_PY_FILE]
```

*   Replace `[PATH_TO_CONFIG_PY_FILE]` with the path to your configuration file (e.g., `src/config/data_prep_config.py` or `src/config/data_prep_configs/my_custom_prep.py`). If omitted, it might try to load a default or raise an error depending on the `load_config` implementation.

The script `src/scripts/prepare_dataset.py` (Lines: 1-32) handles loading the configuration and calling the main preparation logic in `src/data_preparation/prepare_vehicle_dataset.py` (Lines: 193-235).

## Azure ML Execution

To run the data preparation pipeline as an Azure ML job:

1.  **Configure the AML Run Config:** Modify the relevant `.yml` file, for example `deployment/run_configs/dataset_prep.yml`.
    *   Ensure `ENV` points to the correct Azure ML Environment.
    *   Set `EXPERIMENT` to your desired experiment name.
    *   Choose an appropriate `COMPUTE` target.
    *   Under `SCRIPT_ARGUMENTS`, set `CONFIG` to the path *within the Azure ML job context* to your Python configuration file (e.g., `src/config/data_prep_config.py` or `src/config/data_prep_configs/my_custom_prep.py`). If you created a custom config, make sure it's included in the source directory (`SRC`).
    *   Set `SRC` to the source directory (usually ".").
    *   Ensure `ENTRY_SCRIPT` points to `src/scripts/prepare_dataset.py`.

2.  **Submit the Job:** Use the `azure_submit.py` script:
    ```bash
    python azure_submit.py deployment/run_configs/dataset_prep.yml
    ```

This will execute the `src/scripts/prepare_dataset.py` script on the specified Azure ML compute target, using the configuration defined in the Python file pointed to by the `CONFIG` argument in the `.yml` file.
