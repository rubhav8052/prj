#!/bin/bash

# --- Initialize variables ---
PIPELINE_FILE=""
JOB_NAME=""
CONFIG_NAME=""

# --- Parse named arguments ---

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) JOB_NAME="$2"; shift ;;
        --config) CONFIG_NAME="$2"; shift ;;
        --file) PIPELINE_FILE="$2"; shift ;; 
        *) echo "ERROR: Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- VALIDATION: Enforce mandatory arguments ---
# This part remains the same, ensuring name and config are always provided.
if [[ -z "$JOB_NAME" ]] || [[ -z "$CONFIG_NAME" ]] || [[ -z "$PIPELINE_FILE" ]]; then
    echo "ERROR: ( --name , --config or --file) mandatory arguments missing."
    echo ""
    echo "Usage: ./az_submit.sh --name <job-name> --config <config-name> [--file <path/to/pipeline.yml>]"
    exit 1
fi

# --- Check if the specified pipeline file actually exists ---
if [ ! -f "$PIPELINE_FILE" ]; then
    echo "ERROR: Pipeline file not found at '$PIPELINE_FILE'"
    exit 1
fi

# --- If validation passes, execute the dynamic Azure ML command ---
echo "✅ Checks passed. Submitting Azure ML job..."
echo "   Pipeline File: $PIPELINE_FILE"
echo "   Job Name:      $JOB_NAME"
echo "   Config Name:   $CONFIG_NAME"

az ml job create \
  --file "$PIPELINE_FILE" \
  --name "$JOB_NAME" \
  --set inputs.config_name="$CONFIG_NAME"

echo "🚀 Job submission command executed."
