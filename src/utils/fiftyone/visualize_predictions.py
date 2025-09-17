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
Given a ground truth jsonl file and a predictions jsonl files calculates metrices like F1 score, precision and recall.
Provide the crops also, the predictions can be vizualized in Voxel.
"""
import os
import json
import argparse
import fiftyone as fo
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIG ===
LOCAL_GT = "tl_inlay_test_v1_new.jsonl"
LOCAL_PRED = "pred_v1_qwen_new.json"
IMAGE_ROOT = "test_images"
DATASET_NAME = "inlay_dataset"
MIN_IMAGE_HEIGHT = 30 # New constant for minimum image height
DEFAULT_FIFTYONE_PORT = 5151 # Default port for FiftyOne App

def load_and_evaluate(gt_jsonl_path, pred_jsonl_path, image_root, dataset_name="inlay_dataset"):
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    with open(pred_jsonl_path, 'r', encoding='utf-8') as f:
        pred_list = json.load(f)
        predictions = {p["image_name"]: p["output"] for p in pred_list}

    y_true, y_pred = [], []
    skipped_images_count = 0

    # Load ground truth + associate predictions
    with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            image_file = os.path.basename(obj["images"][0])
            image_path = os.path.join(image_root, image_file)

            # Check image height before processing
            try:
                with Image.open(image_path) as img:
                    if img.height < MIN_IMAGE_HEIGHT:
                        skipped_images_count += 1
                        print(f"Skipping image {image_file} due to height ({img.height}px) < {MIN_IMAGE_HEIGHT}px.")
                        continue
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error opening image {image_path}: {e}. Skipping.")
                continue

            gt_label = None
            for convo in reversed(obj.get("conversations", [])):
                if convo["from"] == "assistant":
                    gt_label = convo["value"]
                    break

            sample = fo.Sample(filepath=image_path)

            if gt_label:
                sample["ground_truth"] = fo.Classification(label=gt_label)

            if image_file in predictions:
                pred_label = predictions[image_file]
                sample["prediction"] = fo.Classification(label=pred_label)

                if gt_label:
                    y_true.append(gt_label)
                    y_pred.append(pred_label)

            # Only add to dataset if both ground_truth and prediction are available and not skipped
            if "ground_truth" in sample and "prediction" in sample:
                sample["incorrect_prediction"] = sample["prediction"]["label"] != sample["ground_truth"]["label"]
                dataset.add_sample(sample)

    print(f"\nSkipped {skipped_images_count} images with height below {MIN_IMAGE_HEIGHT} pixels.")

    # Metrics
    if not y_true:
        print("\nNo samples left to evaluate after filtering. Exiting evaluation.")
        return dataset

    print("\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Weighted Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")


    # Confusion matrix
    print("\n🧮 Confusion Matrix:")
    labels = sorted(list(set(y_true).union(set(y_pred))))  # Ensure consistent ordering
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Display matrix with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size for better readability
    disp.plot(xticks_rotation=90, cmap="Blues", ax=ax)

    # Reduce font size and add margins
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Saved confusion matrix as confusion_matrix.png")

    return dataset


def launch_voxel_viewer(dataset, port):
    print(f"🚀 Launching FiftyOne App on port {port}...")
    session = fo.launch_app(dataset, port=port) # Pass the port here
    session.wait()


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions and optionally visualize with FiftyOne")
    parser.add_argument("--vizualize", action="store_true", help="Enable visualization with FiftyOne")
    parser.add_argument("--port", type=int, default=DEFAULT_FIFTYONE_PORT,
                        help=f"Specify the port for the FiftyOne App (default: {DEFAULT_FIFTYONE_PORT})")
    args = parser.parse_args()

    dataset = load_and_evaluate(LOCAL_GT, LOCAL_PRED, IMAGE_ROOT, DATASET_NAME)

    if args.vizualize:
        launch_voxel_viewer(dataset, args.port)
    else:
        print("✅ Done. Visualization skipped.")

if __name__ == "__main__":
    main()