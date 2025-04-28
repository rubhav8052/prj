# Voxel51 Visualization

This document describes how to launch a Voxel51 session, for visualizing datasets or model results, within an Azure ML context.

## Overview

This process uses the FiftyOne library (often imported as `fo`) for data visualization. The setup appears geared towards launching a FiftyOne App session, possibly connected to data within an Azure ML workspace, potentially for interactive debugging or analysis.

*   **Core Library:** FiftyOne
*   **Entry Script:** `src/utils/fiftyone/launch.py` (Based on AML config)
*   **AML Config:** `deployment/run_configs/voxel51.yml`

## Configuration

Configuration is primarily handled through the Azure ML run configuration file (`.yml`).

*   **AML Run Config:** `deployment/run_configs/voxel51.yml`
    *   `ENV`: Specifies the Azure ML Environment (e.g., "fiftyone"). This environment must have `fiftyone` and potentially other necessary libraries installed.
    *   `EXPERIMENT`: Azure ML experiment name.
    *   `COMPUTE`: Compute target to run the session on (can be CPU or GPU depending on needs, CPU often sufficient for visualization).
    *   `RUN_NAME`: Name for the Azure ML run.
    *   `SCRIPT_ARGUMENTS`: Arguments passed to the `launch.py` script.
        *   `source-dir`: specifies a directory containing data or configuration for FiftyOne. Change it according to data you want to visualize.
        *   `timeout`: Session timeout in seconds.
    *   `DEBUG`: Set to `True` to enable debugging features like Jupyter access within the AML job.
    *   `SRC`: Source directory containing the `launch.py` script.
    *   `ENTRY_SCRIPT`: The script to execute (`launch.py`).

*   **Modifying Configuration:**
    1.  Edit the `deployment/run_configs/voxel51.yml` file directly to change the environment, compute, script arguments (like `source-dir` or `timeout`), etc.

## Azure ML Execution

This setup is designed to launch the FiftyOne App within an Azure ML job, making it accessible via the network.

1.  **Ensure Configuration:** Verify the settings in `deployment/run_configs/voxel51.yml`, especially `ENV`, `COMPUTE`, and `SCRIPT_ARGUMENTS`. Ensure `DEBUG` is `True` if you need access to services like Jupyter.
2.  **Submit the Job:**
    ```bash
    python azure_submit.py deployment/run_configs/voxel51.yml
    ```
3.  **Access the Session:**
    *   Once the Azure ML job is running, navigate to the job details page in the Azure ML Studio UI.
    *   Look for the "Details" section for the link. The link will appear after 5 minutes of Job start.

This allows you to interact with the FiftyOne App running on the Azure ML compute resource, accessing data available within that job's context.