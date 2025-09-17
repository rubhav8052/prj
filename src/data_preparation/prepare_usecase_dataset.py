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
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import json
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.data_prep_config import DataPrepConfig
from src.utils.file_utils import save_jsonl

def crop_with_context(image: np.ndarray, bbox: List[int], context_percent: float) -> np.ndarray:
    """Crops an image around a bounding box with added context and padding."""
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    # Compute additional context in pixels
    x_expand = int((x_max - x_min) * context_percent)
    y_expand = int((y_max - y_min) * context_percent)

    # Compute new expanded bounding box
    new_x_min = x_min - x_expand
    new_y_min = y_min - y_expand
    new_x_max = x_max + x_expand
    new_y_max = y_max + y_expand

    # Compute necessary padding if out of bounds
    pad_top = max(0, -new_y_min)
    pad_bottom = max(0, new_y_max - h)
    pad_left = max(0, -new_x_min)
    pad_right = max(0, new_x_max - w)

    # Adjust bbox coordinates for cropping (relative to original image)
    crop_x_min = max(0, new_x_min)
    crop_y_min = max(0, new_y_min)
    crop_x_max = min(w, new_x_max)
    crop_y_max = min(h, new_y_max)

    # Crop the image
    cropped_img = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Add black padding if needed
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped_img = cv2.copyMakeBorder(
            cropped_img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
        )

    return cropped_img

def get_image_path_recursively(base_dir, filename, subfolder=None):
    direct_path = os.path.join(base_dir, filename)
    if os.path.exists(direct_path):
        return direct_path
    if subfolder:
        sub_path = os.path.join(base_dir, subfolder, filename)
        if os.path.exists(sub_path):
            return sub_path
    return None

def prepare_single_image(row_tuple: Tuple[int, pd.Series], config: DataPrepConfig):
    """Loads, processes (draws bbox, crops with context), and saves a single image."""
    index, row = row_tuple
    try:
        
        img_filename = row[config.img_sha_col]
        output_img_filename = f"{row[config.index_col]}.png"
        output_img_path = os.path.join(config.output_dir, 'images', output_img_filename)

        if os.path.exists(output_img_path):
            return
            
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
             img_filename += '.png'

        full_img_path = get_image_path_recursively(
            config.image_base_dir,
            img_filename,
            subfolder=row[config.attribute_col]
        )

        if full_img_path is None:
            print(f"Warning: Image file not found for index {row[config.index_col]}: {img_filename}. Skipping.")
            return

        im = cv2.imread(full_img_path)
        if im is None:
            print(f"Warning: Failed to read image for index {row[config.index_col]}: {full_img_path}. Skipping.")
            return

        # Extract bounding box
        box = np.int0([row[config.bbox_cols[0]], row[config.bbox_cols[1]],
                       row[config.bbox_cols[2]], row[config.bbox_cols[3]]])
        
        if config.min_height_width_pixel:
            if row[config.bbox_cols[2]] - row[config.bbox_cols[0]] < config.min_height_width_pixel or \
                row[config.bbox_cols[3]] - row[config.bbox_cols[1]] < config.min_height_width_pixel:
                print(print(f"Warning:Image too small for index {row[config.index_col]}: {full_img_path}. Skipping."))
                return
        
        # Draw rectangle on the original image (optional, for visualization before crop)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Crop with context using the original image
        processed_img = crop_with_context(im, box, config.context_percent)

        # Save the processed image       
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, processed_img)

    except Exception as e:
        print(f"Error processing image for index {row[config.index_col]} (SHA: {row[config.img_sha_col]}): {e}")


def process_images_threaded(df: pd.DataFrame, config: DataPrepConfig):
    """Processes images in the dataframe using a thread pool."""
    print(f"Processing {len(df)} images using {config.num_threads} threads...")

    def wrapper(row_tuple):
        return prepare_single_image(row_tuple, config)

    with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
        [executor.submit(wrapper, row) for row in df.iterrows()]

    print("Image processing complete.")


def create_jsonl_dataset(df: pd.DataFrame, config: DataPrepConfig, task_type: str, output_filename: str):
    """Creates a JSONL dataset file for a specific task."""
    print(f"Creating {task_type} dataset: {output_filename}...")
    jsonl_data = []
    
    if task_type == config.attribute_col: # Check if the task matches the configured attribute
        prompt_file = config.attribute_prompt_file
        attribute_col_name = config.attribute_col
    else:
        print(f"Warning: Unknown task_type '{task_type}' or mismatch with configured attribute '{config.attribute_col}'. Cannot determine prompt file or attribute column.")
        return

    prompt_content = ""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt_content = f.read().strip()
    elif prompt_file:
        raise ValueError(f"Warning: Prompt file specified but not found: {prompt_file}")
    else:
        raise ValueError(f"Warning: No prompt file specified for task: {task_type}")


    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {task_type} JSONL"):
        image_filename = f"{row[config.index_col]}.png" # Use the unique index for filename
        image_path_relative = './images/'+image_filename

        # Check if the image file actually exists before adding entry
        full_image_path = os.path.join(config.output_dir, 'images', image_filename)
        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found for index {row[config.index_col]}: {full_image_path}. Skipping entry.")
            continue

        # Get the ground truth response from the correct attribute column
        gt_response = row[attribute_col_name]

        entry = {
            "id": f"{task_type}_{row[config.index_col]}",
            "images": [image_path_relative],
            "conversations": [
                {"from": "user", "value": prompt_content},
                {"from": "assistant", "value": str(gt_response)} # Ensure response is string
            ],
            # Optionally include original data for reference
            "original_index": row[config.index_col],
            "original_sha": row[config.img_sha_col],
        }
        jsonl_data.append(entry)

    output_path = os.path.join(config.output_dir, output_filename)
    save_jsonl(jsonl_data, output_path)
    print(f"Saved {task_type} dataset to: {output_path}")


def prepare_datasets(config: DataPrepConfig):
    """Main function to load data, process images, and create JSONL datasets."""
    print("Starting dataset preparation...")
    # --- Load Data ---
    all_dfs = []
    for csv_path in config.input_csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded total {len(df)} rows.")

    # Ensure necessary columns exist
    required_cols = [config.img_sha_col] + config.bbox_cols
    if config.attribute_col not in df.columns:
         print(f"Warning: Configured attribute column '{config.attribute_col}' not found in input CSV(s). Cannot generate attribute dataset.")
    else:
        required_cols.append(config.attribute_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV(s): {missing_cols}")

    # Add index column if it doesn't exist (needed for unique image filenames)
    if config.index_col not in df.columns:
        print(f"'{config.index_col}' column not found, adding sequential index.")
        df[config.index_col] = range(len(df))
    elif not df[config.index_col].is_unique:
        print(f"Warning: '{config.index_col}' column is not unique. Filenames might overwrite.")


    # --- Process Images ---
    os.makedirs(config.output_dir+'/images', exist_ok=True)
    process_images_threaded(df, config) # Assuming this saves images named like "{index}.png"

    # --- Split Data ---
    print("Splitting data into train and validation sets...")
    # Use the index column for splitting to ensure consistency
    indices = df[config.index_col].values
    # Stratify if the attribute column exists and is suitable for stratification
    stratify_col = None
    if config.attribute_col in df.columns:
        # Check if stratification is feasible (e.g., not too many unique values)
        if df[config.attribute_col].nunique() < len(df) / 2: # Heuristic threshold
             stratify_col = df[config.attribute_col]
             print(f"Stratifying split based on column: {config.attribute_col}")
        else:
             print(f"Warning: Column '{config.attribute_col}' has too many unique values for stratification. Splitting without stratification.")

    if config.test_size != 0:
        test_size = config.test_size
        val_size = config.val_size / (1-config.test_size)
        # Split data into train and test
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=config.random_state,
            stratify=stratify_col
        )
    else:
        train_val_indices = indices
        val_size = config.val_size
        test_indices = []

    # Now split train_val into train and val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=config.random_state,
        stratify=stratify_col.loc[df[config.index_col].isin(train_val_indices)] if stratify_col is not None else None
    )

    # Create train/val/test dataframes using the split indices
    X_train = df[df[config.index_col].isin(train_indices)].copy()
    X_val = df[df[config.index_col].isin(val_indices)].copy()
    X_test = df[df[config.index_col].isin(test_indices)].copy()

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # --- Create JSONL Datasets ---
    # Use the configured attribute column and filenames
    if config.attribute_col in df.columns:
        create_jsonl_dataset(X_train, config, config.attribute_col, config.attribute_train_file)
        create_jsonl_dataset(X_val, config, config.attribute_col, config.attribute_val_file)
        if len(X_test) != 0:
            create_jsonl_dataset(X_test, config, config.attribute_col, config.attribute_test_file)
    else:
        print(f"Skipping attribute dataset creation: column '{config.attribute_col}' not found.")

    print("Dataset preparation finished.") 