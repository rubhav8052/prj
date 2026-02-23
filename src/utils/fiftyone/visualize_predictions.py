# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================

"""
Unified Evaluation Tool

This script automatically adapts to evaluate different prediction types:
- Right-of-Way (ROW) scenarios with multi-directional attributes
- Traffic Light Inlays (single classification task)

The script detects the data structure and applies the appropriate 
evaluation method without requiring explicit configuration.

Features:
- Auto-detection of data type and evaluation metrics
- Quantitative evaluation with appropriate metrics for each use case
- Optional qualitative visualization in FiftyOne/Voxel
- Export of evaluation results and confusion matrices

Usage:
    python unified_eval.py                      # Run evaluation only
    python unified_eval.py --visualize          # Run evaluation and launch visualization
    python unified_eval.py --gt PATH --pred PATH --images PATH  # Custom data paths
"""
import os
import json
import argparse
import fiftyone as fo
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

# === DEFAULT CONFIG ===
DEFAULT_FIFTYONE_PORT = 5151
MIN_IMAGE_HEIGHT = 30  # Minimum image height for filtering - applicable only for TL usecase

# Default paths can be overridden with command-line arguments
DEFAULT_GT = "/home/fss2kor/dsp-op-attribute-labeling/gt.jsonl"
DEFAULT_PRED = "/home/fss2kor/dsp-op-attribute-labeling/pred.json"
DEFAULT_IMAGES = "/home/fss2kor/dsp-op-attribute-labeling/images/test_images"
DEFAULT_DATASET = "evaluation_dataset"

def is_row_data(gt_path, pred_path):
    """
    Detect if the data follows ROW format (with direction attributes) 
    or single-class inlay format
    
    Returns:
        bool: True if ROW format, False if inlay format
    """
    # First check prediction format
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
            if isinstance(pred_data, list) and len(pred_data) > 0:
                # Check first prediction
                sample_pred = pred_data[0]
                if "output" in sample_pred:
                    output = sample_pred["output"]
                    # If output is string that can be parsed as JSON with direction keys
                    if isinstance(output, str):
                        try:
                            parsed = json.loads(output)
                            if any(key in parsed for key in ["left", "right", "straight"]):
                                return True
                        except json.JSONDecodeError:
                            # If we can't parse it as JSON, likely a simple string classification
                            return False
    except Exception:
        pass
    
    # If prediction check is inconclusive, check ground truth
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    for convo in reversed(obj.get("conversations", [])):
                        if convo["from"] == "assistant" and "value" in convo:
                            value = convo["value"]
                            # If value is string that can be parsed as JSON with direction keys
                            if isinstance(value, str):
                                try:
                                    parsed = json.loads(value)
                                    if isinstance(parsed, dict) and any(key in parsed for key in ["left", "right", "straight"]):
                                        return True
                                except json.JSONDecodeError:
                                    # If value is just a simple string, it's inlay format
                                    return False
                            # If value is already a dict with direction keys
                            elif isinstance(value, dict) and any(key in value for key in ["left", "right", "straight"]):
                                return True
                            else:
                                return False
                    break
    except Exception:
        pass
    
    # Default to False (inlay) if detection fails
    return False

def load_and_evaluate(gt_jsonl_path, pred_jsonl_path, image_root, dataset_name=DEFAULT_DATASET):
    """
    Unified function that automatically detects data type and applies appropriate evaluation.
    
    Returns:
        fo.Dataset: The created dataset with ground truth and predictions.
    """
    # Detect if this is ROW or inlay data
    is_row = is_row_data(gt_jsonl_path, pred_jsonl_path)
    eval_type = "Right of Way" if is_row else "TL Inlay"
    print(f"\nAuto-detected usecase data type: {eval_type}")
    
    # Clean up existing dataset if it exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    
    # Initialize metrics variables
    skipped_images = 0
    missing_images = 0
    
    # Load predictions into a lookup dictionary
    with open(pred_jsonl_path, 'r', encoding='utf-8') as f:
        pred_list = json.load(f)
        if is_row:
            predictions = {p["image_name"]: json.loads(p["output"]) for p in pred_list}
        else:
            predictions = {p["image_name"]: p["output"] for p in pred_list}
    
    if is_row:
        # === ROW EVALUATION ===
        y_true, y_pred = {"left": [], "right": [], "straight": []}, {"left": [], "right": [], "straight": []}
        total_images = 0
        correct_images = 0
        total_predictions = 0
        correct_predictions = 0
        
        # Track per-direction performance
        direction_stats = {
            "left": {"total": 0, "correct": 0},
            "right": {"total": 0, "correct": 0},
            "straight": {"total": 0, "correct": 0}
        }
        
        # Process ground truth data
        with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                    image_file = os.path.basename(obj["images"][0])
                    image_path = os.path.join(image_root, image_file)
                    
                    # Extract ground truth labels
                    gt_labels = {}
                    for convo in reversed(obj.get("conversations", [])):
                        if convo["from"] == "assistant" and "value" in convo:
                            try:
                                gt_labels = json.loads(convo["value"]) if isinstance(convo["value"], str) else convo["value"]
                                break
                            except json.JSONDecodeError:
                                print(f"WARNING: Cannot parse GT labels for image {image_file}")
                    
                    # Check if we can add this to FiftyOne visualization
                    can_visualize = True
                    
                    if not os.path.exists(image_path):
                        print(f"WARNING: Image not found: {image_path}")
                        missing_images += 1
                        can_visualize = False
                    else:
                        # Check image height before processing
                        try:
                            with Image.open(image_path) as img:
                                if img.height < MIN_IMAGE_HEIGHT:
                                    print(f"WARNING: Skipping image {image_file} from visualization (height: {img.height}px < {MIN_IMAGE_HEIGHT}px)")
                                    skipped_images += 1
                                    can_visualize = False
                        except Exception as e:
                            print(f"WARNING: Error checking image {image_path}: {str(e)}")
                            can_visualize = False

                    # Process predictions if available - regardless of image availability
                    if image_file in predictions and gt_labels:
                        pred_labels = predictions[image_file]
                        
                        # Create a FiftyOne sample if we can visualize
                        if can_visualize:
                            sample = fo.Sample(filepath=image_path)
                            sample["gt_raw"] = gt_labels
                            
                            gt_text = "Ground Truth:\n"
                            for direction in ["left", "right", "straight"]:
                                if direction in gt_labels:
                                    gt_text += f"{direction}: {gt_labels[direction]}\n"
                            sample["ground_truth_text"] = gt_text.strip()
                            
                            sample["pred_raw"] = pred_labels
                            
                            pred_text = "Predictions:\n"
                            for direction in ["left", "right", "straight"]:
                                if direction in pred_labels:
                                    pred_text += f"{direction}: {pred_labels[direction]}\n"
                            sample["prediction_text"] = pred_text.strip()

                            # Create classifications for visualization
                            gt_classifications = []
                            pred_classifications = []
                            
                            for direction in ["left", "right", "straight"]:
                                if direction in gt_labels and direction in pred_labels:
                                    # Format labels for display
                                    gt_label = f"{direction}: {gt_labels[direction]}"
                                    pred_label = f"{direction}: {pred_labels[direction]}"
                                    
                                    gt_classifications.append(fo.Classification(label=gt_label))
                                    pred_classifications.append(fo.Classification(label=pred_label))

                            # Add classification data to sample
                            sample["ground_truth"] = fo.Classifications(classifications=gt_classifications)
                            sample["prediction"] = fo.Classifications(classifications=pred_classifications)
                        
                        # Evaluate prediction accuracy - do this regardless of image availability
                        all_correct = True
                        for direction in ["left", "right", "straight"]:
                            if direction in gt_labels and direction in pred_labels:
                                gt_value = gt_labels[direction]
                                pred_value = pred_labels[direction]
                                
                                # Track metrics
                                y_true[direction].append(gt_value)
                                y_pred[direction].append(pred_value)
                                total_predictions += 1
                                direction_stats[direction]["total"] += 1
                                
                                if gt_value == pred_value:
                                    correct_predictions += 1
                                    direction_stats[direction]["correct"] += 1
                                else:
                                    all_correct = False

                        if can_visualize:
                            sample["incorrect_prediction"] = not all_correct
                            dataset.add_sample(sample)

                        total_images += 1
                        if all_correct:
                            correct_images += 1
                    elif can_visualize:
                        # Add sample to dataset even if no predictions available
                        sample = fo.Sample(filepath=image_path)
                        if gt_labels:
                            sample["gt_raw"] = gt_labels
                            gt_text = "Ground Truth:\n"
                            for direction in ["left", "right", "straight"]:
                                if direction in gt_labels:
                                    gt_text += f"{direction}: {gt_labels[direction]}\n"
                            sample["ground_truth_text"] = gt_text.strip()
                            
                            gt_classifications = []
                            for direction in ["left", "right", "straight"]:
                                if direction in gt_labels:
                                    gt_label = f"{direction}: {gt_labels[direction]}"
                                    gt_classifications.append(fo.Classification(label=gt_label))
                        
                        dataset.add_sample(sample)
                    
                except Exception as e:
                    print(f"WARNING: Error processing line {line_idx}: {str(e)}")

        # Calculate performance metrics
        if total_predictions == 0:
            print("WARNING: No valid predictions found. Check your data and paths.")
            return dataset
            
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        per_image_accuracy = correct_images / total_images if total_images > 0 else 0
        
        # Calculate per-direction accuracy
        direction_accuracy = {}
        for direction in ["left", "right", "straight"]:
            stats = direction_stats[direction]
            direction_accuracy[direction] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        # Print summary results
        print("\nEvaluation Summary:")
        print(f"Total Images: {total_images}")
        if missing_images > 0:
            print(f"Missing Images: {missing_images} (evaluated but not visualized)")
        if skipped_images > 0:
            print(f"Skipped Images: {skipped_images} (too small for visualization)")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Per-Image Accuracy: {per_image_accuracy:.4f}")
        print("\nPer-Direction Accuracy:")
        for direction, accuracy in direction_accuracy.items():
            stats = direction_stats[direction]
            print(f"  {direction}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")

        # Generate confusion matrix
        combined_y_true = []
        combined_y_pred = []
        for direction in ["left", "right", "straight"]:
            combined_y_true.extend(y_true[direction])
            combined_y_pred.extend(y_pred[direction])

        # Save confusion matrix without displaying
        classes = sorted(set(combined_y_true))
        cm = confusion_matrix(combined_y_true, combined_y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Total Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close(fig)
        print("Saved confusion matrix as confusion_matrix.png")
        
        # Export detailed metrics
        metrics = {
            "evaluation_type": "row",
            "overall_metrics": {
                "total_images": total_images,
                "missing_images": missing_images,
                "skipped_images": skipped_images,
                "correct_images": correct_images,
                "per_image_accuracy": per_image_accuracy,
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "overall_accuracy": overall_accuracy
            },
            "per_direction_metrics": {
                direction: {
                    "total": direction_stats[direction]["total"],
                    "correct": direction_stats[direction]["correct"],
                    "accuracy": direction_accuracy[direction]
                } for direction in ["left", "right", "straight"]
            },
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm.tolist()
            }
        }
        
    else:
        # === INLAY EVALUATION ===
        y_true, y_pred = [], []
        
        # Process ground truth data
        with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)
                    image_file = os.path.basename(obj["images"][0])
                    image_path = os.path.join(image_root, image_file)
                    
                    # Extract ground truth label
                    gt_label = None
                    for convo in reversed(obj.get("conversations", [])):
                        if convo["from"] == "assistant":
                            gt_label = convo["value"] if "value" in convo else convo.get("content")
                            break

                    # Check if we can add this to FiftyOne visualization
                    can_visualize = True
                    
                    if not os.path.exists(image_path):
                        print(f"WARNING: Image not found: {image_path}")
                        missing_images += 1
                        can_visualize = False
                    else:
                        # Check image height before processing
                        try:
                            with Image.open(image_path) as img:
                                if img.height < MIN_IMAGE_HEIGHT:
                                    print(f"WARNING: Skipping image {image_file} from visualization (height: {img.height}px < {MIN_IMAGE_HEIGHT}px)")
                                    skipped_images += 1
                                    can_visualize = False
                        except Exception as e:
                            print(f"WARNING: Error checking image {image_path}: {str(e)}")
                            can_visualize = False

                    # Process for evaluation regardless of image availability
                    if image_file in predictions and gt_label:
                        pred_label = predictions[image_file]
                        y_true.append(gt_label)
                        y_pred.append(pred_label)
                        
                        # Add to FiftyOne dataset if we can visualize
                        if can_visualize:
                            sample = fo.Sample(filepath=image_path)
                            sample["ground_truth"] = fo.Classification(label=gt_label)
                            sample["prediction"] = fo.Classification(label=pred_label)
                            sample["incorrect_prediction"] = gt_label != pred_label
                            dataset.add_sample(sample)
                    elif can_visualize and gt_label:
                        # Add sample to dataset even if no predictions available
                        sample = fo.Sample(filepath=image_path)
                        sample["ground_truth"] = fo.Classification(label=gt_label)
                        dataset.add_sample(sample)
                    
                except Exception as e:
                    print(f"WARNING: Error processing line {line_idx}: {str(e)}")

        # Calculate metrics
        if not y_true:
            print("WARNING: No valid predictions found. Check your data and paths.")
            return dataset

        # Generate classification report
        report = classification_report(y_true, y_pred, digits=3, zero_division=0)
        
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Print results
        print("\nEvaluation Summary:")
        print(f"Total Images: {len(y_true)}")
        if missing_images > 0:
            print(f"Missing Images: {missing_images} (evaluated but not visualized)")
        if skipped_images > 0:
            print(f"Skipped Images: {skipped_images} (too small for visualization)")
        print(f"\nClassification Report:\n{report}")
        print(f"Macro Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"Weighted Precision: {weighted_precision:.3f}, Recall: {weighted_recall:.3f}, F1: {weighted_f1:.3f}")

        # Generate confusion matrix
        labels = sorted(list(set(y_true).union(set(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(xticks_rotation=90, cmap="Blues", ax=ax)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close(fig)
        print("Saved confusion matrix as confusion_matrix.png")
        
        # Export metrics
        metrics = {
            "evaluation_type": "inlay",
            "overall_metrics": {
                "total_images": len(y_true),
                "missing_images": missing_images,
                "skipped_images": skipped_images,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1
            },
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist()
            },
            "classification_report": report
        }
    
    # Save metrics to file
    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved detailed metrics to evaluation_results.json")

    return dataset


def launch_voxel_viewer(dataset, port=DEFAULT_FIFTYONE_PORT):
    """Launch the FiftyOne App for visualization."""
    print(f"Launching FiftyOne App on port {port}...")
    session = fo.launch_app(dataset, port=port)
    session.wait()


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation tool")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization with FiftyOne")
    parser.add_argument("--port", type=int, default=DEFAULT_FIFTYONE_PORT,
                        help=f"Port for FiftyOne App (default: {DEFAULT_FIFTYONE_PORT})")
    parser.add_argument("--gt", help="Path to ground truth file")
    parser.add_argument("--pred", help="Path to predictions file")
    parser.add_argument("--images", help="Path to image directory")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Name for the FiftyOne dataset")
    args = parser.parse_args()

    # Use provided paths or defaults
    gt_path = args.gt or DEFAULT_GT
    pred_path = args.pred or DEFAULT_PRED
    image_path = args.images or DEFAULT_IMAGES
    dataset_name = args.dataset
    
    print(f"\nGround Truth: {gt_path}")
    print(f"Predictions: {pred_path}")
    print(f"Images: {image_path}")
    
    # Run evaluation with automatic data type detection
    dataset = load_and_evaluate(gt_path, pred_path, image_path, dataset_name)

    if args.visualize:
        launch_voxel_viewer(dataset, args.port)
    else:
        print("Done. Visualization skipped.")


if __name__ == "__main__":
    main()