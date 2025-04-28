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
import sys
import json
from tqdm import tqdm
from typing import List, Dict, Any


from swift.llm import (
    PtEngine, RequestConfig as SwiftInferRequestConfig, safe_snapshot_download,
    get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

from src.config.swift_inference_config import SwiftInferenceConfig, SwiftRequestConfig
from src.utils.file_utils import load_jsonl, save_jsonl, get_latest_checkpoint_path

def run_swift_inference(config: SwiftInferenceConfig):
    """Runs inference using Swift PtEngine based on the provided configuration."""

    print(f"\n Running Swift inference with config:")
    print(f"   Model ID: {config.model_id}")
    print(f"   LoRA Path: {config.lora_path}")
    print(f"   Dataset: {config.dataset_path}")
    print(f"   Output Dir: {config.output_dir}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Dtype: {config.dtype}")

    output_file = os.path.join(config.output_dir, f"{config.output_filename_prefix}_output.jsonl")
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Skipping inference.")
        return
    os.makedirs(config.output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    model_id_or_path = config.model_id

    # Handle LoRA path - potentially download if it's a model ID/path
    actual_lora_path = None
    if config.lora_path:
        # If lora_path points to a directory containing checkpoints, find the latest
        if 'checkpoint-' in config.lora_path and os.path.isdir(os.path.dirname(config.lora_path)):
             print(f"LoRA path seems to be a specific checkpoint: {config.lora_path}")
             actual_lora_path = config.lora_path # Assume it's the correct one
        elif os.path.isdir(config.lora_path):
             print(f"LoRA path is a directory, searching for latest checkpoint in: {config.lora_path}")
             latest_ckpt = get_latest_checkpoint_path(config.lora_path)
             if latest_ckpt:
                 actual_lora_path = latest_ckpt
                 # Try to load args.json to potentially get base model if not specified
                 args_path = os.path.join(latest_ckpt, 'args.json')
                 if os.path.exists(args_path):
                     try:
                         with open(args_path, 'r') as f:
                             ckpt_args = json.load(f)
                             # If base model in config is different, warn or override?
                             if 'model' in ckpt_args and ckpt_args['model'] != config.model_id:
                                 print(f"Warning: Model ID in config ({config.model_id}) differs from checkpoint args ({ckpt_args['model']}). Using config model ID.")
                             # Potentially load lora_target_modules if needed/available
                             lora_target_modules = ckpt_args.get('lora_target_modules')
                             if not config.template_type and 'template_type' in ckpt_args:
                                 config.template_type = ckpt_args['template_type']
                                 print(f"Using template type from checkpoint args: {config.template_type}")

                     except Exception as e:
                         print(f"Warning: Could not read args.json from checkpoint {latest_ckpt}: {e}")
             else:
                 print(f"Warning: No checkpoint found in LoRA path directory: {config.lora_path}. Trying path directly.")
                 # Attempt to use the path directly if it might be a model ID on hub
                 actual_lora_path = safe_snapshot_download(config.lora_path)

        else:
             # Assume lora_path is a model ID or path needing download
             print(f"Attempting to download LoRA adapter from: {config.lora_path}")
             actual_lora_path = safe_snapshot_download(config.lora_path)


    model, tokenizer = get_model_tokenizer(model_id_or_path) # Pass dtype here

    if actual_lora_path:
        print(f"Loading LoRA adapter from: {actual_lora_path}")
        try:
            # Pass dtype explicitly if needed, though from_pretrained might handle it
            model = Swift.from_pretrained(model, actual_lora_path, dtype=config.dtype, inference_mode=True)
            print("Successfully loaded LoRA adapter.")
        except Exception as e:
            print(f"Error loading LoRA adapter from {actual_lora_path}: {e}")
            print("Proceeding without LoRA adapter.")
            # Decide whether to raise error or continue with base model
            # raise e
    else:
        print("No LoRA path provided or loaded. Using base model.")


    # --- Setup Template and Engine ---
    template_type = config.template_type or getattr(getattr(model, 'model_meta', None), 'template', None) # Infer if not set
    if not template_type:
         print("Warning: Could not determine template type. Using default.")
         # Swift might handle default, or specify one like 'default-generation' if needed
         template_type = 'default-generation' # Example fallback

    print(f"Using template: {template_type}")
    template = get_template(template_type, tokenizer)

    print("Initializing PtEngine...")
    # Ensure model is on the correct device (PtEngine might handle this)
    # model = model.cuda() # Or let PtEngine manage device placement
    engine = PtEngine.from_model_template(model, template, max_batch_size=config.batch_size)
    print("PtEngine initialized.")

    # --- Prepare Request Configuration ---
    swift_req_config = SwiftInferRequestConfig(
        max_tokens=config.request_config.max_tokens,
        temperature=config.request_config.temperature,
        top_p=config.request_config.top_p,
        top_k=config.request_config.top_k,
        repetition_penalty=config.request_config.repetition_penalty,
    )
    print(f"Using Request Config: {swift_req_config}")

    # --- Load Dataset ---
    print(f"Loading dataset from: {config.dataset_path}")
    try:
        dataset = load_jsonl(config.dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.dataset_path}")
        return
    except Exception as e:
        print(f"Error loading dataset {config.dataset_path}: {e}")
        return

    if not dataset:
        print("Error: Dataset is empty.")
        return

    print(f"Loaded {len(dataset)} entries from dataset.")
    results = []

    # --- Run Inference Loop ---
    print("Starting inference...")
    for i in tqdm(range(0, len(dataset), config.batch_size), desc="Inference Batches"):
        batch = dataset[i:i + config.batch_size]
        infer_requests = []
        batch_entries = []

        for entry in batch:
            # Ensure query and images keys exist
            query = entry.get('conversations')[0]['value']
            images = entry.get('images', []) # Default to empty list if missing
            gt_response = entry.get("response", entry.get("gt_reponse")) # Get ground truth

            if query is None:
                print(f"Warning: Skipping entry due to missing 'query': {entry}")
                continue


            messages = [{'role': 'user', 'content': query}]
            infer_requests.append(InferRequest(
                messages=messages,
                images=images,
                # Store original entry to easily retrieve GT response later
            ))
            batch_entries.append(entry)

        if not infer_requests:
            continue

        try:
            responses = engine.infer(infer_requests, swift_req_config)
        except Exception as e:
            print(f"\nError during engine.infer on batch starting at index {i}: {e}")
            responses = [] 

        # Process responses
        for req, resp, entry in zip(infer_requests, responses, batch_entries):
            if resp.choices:
                predicted_response = resp.choices[0].message.content
            else:
                print(f"Warning: No response generated for request. Error: {resp.error}")
                predicted_response = "[ERROR]"

            # Retrieve original entry from request_id
            original_entry = entry

            result = {
                "query": original_entry.get('query'),
                "gt_reponse": original_entry.get("response", original_entry.get("gt_reponse")), # Get GT from original
                "response": predicted_response,
                "images": original_entry.get('images', [])
            }
            results.append(result)

    # --- Save Outputs ---
    print(f"Inference complete. Saving {len(results)} results...")
    save_jsonl(results, output_file)
    print(f"✅ Output saved to: {output_file}") 