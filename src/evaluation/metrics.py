# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================

import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, confusion_matrix,
    accuracy_score, classification_report
)
import matplotlib.ticker as mtick
from typing import Dict, List, Any, Optional, Tuple

from src.config.evaluation_config import EvaluationConfig
from src.utils.file_utils import load_all_inference_outputs # Assuming load_all_jsons is renamed

sns.set(style="whitegrid")

def extract_occlusion(text: str) -> Optional[float]:
    """Extracts occlusion percentage from text."""
    if isinstance(text, (int, float)): # Handle cases where it might already be numeric
        return float(text)

    # Try extracting percentage first
    match_percent = re.search(r'(\d+(\.\d+)?)\s*%', text)
    if match_percent:
        return float(match_percent.group(1))

    # If no '%', try extracting any number
    match_num = re.search(r'\d+(\.\d+)?', text)
    if match_num:
        # Check if the number looks like a percentage (0-100)
        val = float(match_num.group(0))
        if 0 <= val <= 100:
             # print(f"Warning: Extracted number '{val}' without '%' sign. Assuming it's a percentage.")
             return val
        else:
             print(f"Warning: Extracted number '{val}' is outside 0-100 range. Cannot interpret as occlusion. Text: '{text}'")
             return None # Or handle differently if non-percentage numbers are expected

    # If it's just a number string without %
    try:
        val = float(text)
        if 0 <= val <= 100:
            # print(f"Warning: Interpreting plain number '{val}' as percentage. Text: '{text}'")
            return val
        else:
            print(f"Warning: Plain number '{val}' is outside 0-100 range. Cannot interpret as occlusion. Text: '{text}'")
            return None
    except ValueError:
        print(f"Warning: Could not extract occlusion value from text: '{text}'")
        return None


def is_classification(entry: Dict[str, Any]) -> bool:
    """Infers if the task is classification based on ground truth format."""
    gt_response = entry.get("gt_reponse", entry.get("response")) # Check both possible keys
    if gt_response is None:
        print("Warning: Cannot determine task type, 'gt_reponse' or 'response' key missing.")
        return False # Default or raise error?

    # Simple heuristic: if GT is numeric-like and potentially a percentage, assume regression (occlusion).
    # Otherwise, assume classification.
    if isinstance(gt_response, (int, float)):
        return False # Likely regression/occlusion
    if isinstance(gt_response, str):
        # Try extracting occlusion. If successful, it's likely occlusion.
        if extract_occlusion(gt_response) is not None:
             # Check if it ONLY contains the number/percentage, otherwise could be classification text
             cleaned_gt = re.sub(r'(\d+(\.\d+)?)\s*%', '', gt_response).strip()
             if not cleaned_gt: # If nothing left after removing percentage, likely occlusion
                 return False
             else: # Contains other text, likely classification
                 return True
        else:
            # Cannot extract occlusion, likely classification
            return True
    return True # Default to classification for other types


def evaluate_occlusion(model_name: str, data: List[Dict[str, Any]], config: EvaluationConfig) -> Dict[str, float]:
    """Evaluates occlusion estimation performance."""
    print(f"Evaluating occlusion for model: {model_name}")
    gt, pred = [], []
    parse_errors = 0

    for entry in data:
        gt_val = extract_occlusion(entry.get("gt_reponse", entry.get("response"))) # Check both keys
        pred_val = extract_occlusion(entry.get("response"))

        if gt_val is not None and pred_val is not None:
            gt.append(gt_val)
            pred.append(pred_val)
        else:
            parse_errors += 1
            print(f"Debug: Could not parse entry: GT='{entry.get('gt_reponse')}', Pred='{entry.get('response')}'")


    if parse_errors > 0:
        print(f"Warning: Skipped {parse_errors}/{len(data)} entries due to parsing errors.")

    if not gt or not pred:
        print("Error: No valid ground truth or prediction values found for occlusion evaluation.")
        return {"mae": float('nan'), "rmse": float('nan')}

    gt = np.array(gt)
    pred = np.array(pred)

    mae = mean_absolute_error(gt, pred)
    rmse = np.sqrt(mean_squared_error(gt, pred)) # Calculate RMSE correctly

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Ensure output directory exists
    os.makedirs(config.evaluation_output_dir, exist_ok=True)

    # --- Scatter Plot ---
    plt.figure(figsize=(7, 5)) # Adjusted size slightly
    plt.scatter(gt, pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([0, 100], [0, 100], '--r', label='Ideal Fit')
    plt.xlabel("Ground Truth Occlusion (%)")
    plt.ylabel("Predicted Occlusion (%)")
    plt.title(f"Occlusion Estimation: {model_name}\nMAE={mae:.2f} | RMSE={rmse:.2f}")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(config.evaluation_output_dir, f"{model_name}_occlusion_scatter.png")
    plt.savefig(scatter_path)
    print(f"Saved scatter plot to {scatter_path}")
    # plt.show() # Avoid showing plots in script mode unless intended
    plt.close()

    # --- Sorted Confusion Matrix (Binned) ---
    bin_size = config.occlusion_bin_size
    bins = list(range(0, 101, bin_size)) # Ensure bins go up to 100 inclusive
    # Adjust labels to be more precise, e.g., [0-10), [10-20), ..., [90-100]
    labels = [f"[{i}-{i+bin_size})" for i in bins[:-2]] + [f"[{bins[-2]}-{bins[-1]}]"] # Last bin inclusive

    # Handle potential NaN or out-of-range values before binning
    valid_indices = ~np.isnan(gt) & ~np.isnan(pred)
    gt_valid = gt[valid_indices]
    pred_valid = pred[valid_indices]

    # Clip values to be within [0, 100] before binning to avoid issues with edges
    gt_clipped = np.clip(gt_valid, 0, 100)
    pred_clipped = np.clip(pred_valid, 0, 100)

    # Use include_lowest=True for the first bin if starting at 0
    gt_bins = pd.cut(gt_clipped, bins=bins, labels=labels, right=False, include_lowest=True)
    pred_bins = pd.cut(pred_clipped, bins=bins, labels=labels, right=False, include_lowest=True)

    # Handle potential NaNs introduced by pd.cut if values fall exactly on bin edges weirdly
    gt_bins = gt_bins.astype(str)
    pred_bins = pred_bins.astype(str)


    # Calculate confusion matrix using crosstab (more robust for categorical)
    # Normalize by index (ground truth) to show prediction distribution for each true bin
    cm = pd.crosstab(gt_bins, pred_bins, rownames=['Ground Truth'], colnames=['Prediction'], dropna=False, normalize='index') * 100 # Multiply by 100 for percentage

    # Ensure all potential labels are present, fill missing with 0
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)


    plt.figure(figsize=(9, 7)) # Adjusted size
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'format': '%.0f%%', 'label': '% Predictions'})
    plt.title(f"Occlusion Confusion Matrix (%): {model_name}\n(Rows Normalized)")
    plt.xlabel("Predicted Occlusion Bin")
    plt.ylabel("Ground Truth Occlusion Bin")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(config.evaluation_output_dir, f"{model_name}_occlusion_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    # plt.show()
    plt.close()

    return {
        "mae": mae,
        "rmse": rmse
    }


def evaluate_vehicle_type(model_name: str, data: List[Dict[str, Any]], config: EvaluationConfig) -> Dict[str, Any]:
    """Evaluates vehicle type classification performance."""
    print(f"Evaluating vehicle type for model: {model_name}")
    gt, pred = [], []
    norm_map = config.class_normalization_map or {}

    for entry in data:
        gt_val = entry.get("gt_reponse", entry.get("response", "")).strip().lower() # Check both keys
        pred_val = entry.get("response", "").strip().lower()

        # Normalize classes if map provided
        gt_val = norm_map.get(gt_val, gt_val)
        pred_val = norm_map.get(pred_val, pred_val)

        if gt_val: # Only add if ground truth is not empty
            gt.append(gt_val)
            pred.append(pred_val if pred_val else "[empty]") # Handle empty predictions
        else:
            print(f"Warning: Skipping entry with empty ground truth: {entry}")


    if not gt:
        print("Error: No valid ground truth values found for classification evaluation.")
        return {"accuracy": float('nan'), "report": {}, "labels": []}

    # Determine labels
    if config.classification_labels:
        labels = config.classification_labels
    else:
        labels = sorted(list(set(gt + pred)))

    acc = accuracy_score(gt, pred)
    # Use labels parameter in classification_report and confusion_matrix
    report = classification_report(gt, pred, labels=labels, output_dict=True, zero_division=0)
    conf = confusion_matrix(gt, pred, labels=labels)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(gt, pred, labels=labels, zero_division=0))


    df_cm = pd.DataFrame(conf, index=labels, columns=labels)

    # Ensure output directory exists
    os.makedirs(config.evaluation_output_dir, exist_ok=True)

    # Save Confusion Matrix CSV
    cm_csv_path = os.path.join(config.evaluation_output_dir, f"{model_name}_vehicle_type_confusion_matrix.csv")
    df_cm.to_csv(cm_csv_path)
    print(f"Saved confusion matrix CSV to {cm_csv_path}")


    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(len(labels)*0.8 + 2, len(labels)*0.7 + 1)) # Dynamic sizing
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=.5)
    plt.title(f"Vehicle Type Confusion Matrix: {model_name}\nAccuracy={acc:.2%}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_plot_path = os.path.join(config.evaluation_output_dir, f"{model_name}_vehicle_type_confusion_heatmap.png")
    plt.savefig(cm_plot_path)
    print(f"Saved confusion matrix heatmap to {cm_plot_path}")
    # plt.show()
    plt.close()

    return {
        "accuracy": acc,
        "report": report,
        "labels": labels # Return labels used
    }


def run_evaluation(config: EvaluationConfig):
    """Loads inference outputs and runs evaluation based on config."""
    print(f"Starting evaluation for results in: {config.inference_output_folder}")
    model_data = load_all_inference_outputs(config.inference_output_folder)

    if not model_data:
        print("Error: No model inference data loaded. Exiting evaluation.")
        return

    summary = []

    for model_name, entries in model_data.items():
        if not entries:
            print(f"Warning: No entries found for model {model_name}. Skipping.")
            continue

        # Determine task type
        task_type = config.task_type
        if task_type == 'auto':
            try:
                task_type = "classification" if is_classification(entries[0]) else "occlusion"
                print(f"Inferred task type for {model_name}: {task_type}")
            except Exception as e:
                print(f"Error inferring task type for {model_name}: {e}. Skipping model.")
                continue

        metrics = {"model": model_name, "task": task_type}
        try:
            if task_type == "occlusion":
                occlusion_metrics = evaluate_occlusion(model_name, entries, config)
                metrics.update(occlusion_metrics)
            elif task_type == "classification":
                classification_metrics = evaluate_vehicle_type(model_name, entries, config)
                # Flatten report for easier CSV storage if needed, or keep as dict
                metrics["accuracy"] = classification_metrics["accuracy"]
                # Add precision, recall, f1 for macro/weighted avg
                metrics["macro_avg_precision"] = classification_metrics["report"]["macro avg"]["precision"]
                metrics["macro_avg_recall"] = classification_metrics["report"]["macro avg"]["recall"]
                metrics["macro_avg_f1"] = classification_metrics["report"]["macro avg"]["f1-score"]
                metrics["weighted_avg_precision"] = classification_metrics["report"]["weighted avg"]["precision"]
                metrics["weighted_avg_recall"] = classification_metrics["report"]["weighted avg"]["recall"]
                metrics["weighted_avg_f1"] = classification_metrics["report"]["weighted avg"]["f1-score"]
                # Optionally store the full report dict as JSON string or similar if needed in CSV
                # metrics["full_report"] = json.dumps(classification_metrics["report"])
            else:
                print(f"Warning: Unsupported task type '{task_type}' for model {model_name}. Skipping.")
                continue

            summary.append(metrics)

        except Exception as e:
            print(f"Error during evaluation for model {model_name}: {e}")
            # Add partial results if possible, or skip
            summary.append({"model": model_name, "task": task_type, "error": str(e)})


    if not summary:
        print("No evaluation results generated.")
        return

    # Create and save summary DataFrame
    df_summary = pd.DataFrame(summary)

    # Reorder columns for clarity
    cols = ["model", "task"]
    if any(m.get("task") == "occlusion" for m in summary):
        cols.extend(["mae", "rmse"])
    if any(m.get("task") == "classification" for m in summary):
        cols.extend([
            "accuracy", "macro_avg_precision", "macro_avg_recall", "macro_avg_f1",
            "weighted_avg_precision", "weighted_avg_recall", "weighted_avg_f1"
        ])
    cols.extend([c for c in df_summary.columns if c not in cols]) # Add any remaining columns
    df_summary = df_summary[cols]


    print("\n=== EVALUATION SUMMARY ===")
    print(df_summary.to_string())

    # Save summary table
    os.makedirs(config.evaluation_output_dir, exist_ok=True)
    summary_path = os.path.join(config.evaluation_output_dir, config.summary_csv_filename)
    df_summary.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Evaluation summary saved to: {summary_path}") 