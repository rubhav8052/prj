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
import glob
from typing import List, Dict, Any, Optional

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_latest_checkpoint_path(base_path: str) -> Optional[str]:
    """Finds the latest checkpoint directory within a base path."""
    candidates = glob.glob(os.path.join(base_path, '**', 'checkpoint-*'), recursive=True)
    if not candidates:
        print(f"Warning: No checkpoints found under {base_path}")
        return None
    try:
        # Ensure only directories are considered and extract step number
        valid_checkpoints = {}
        for candidate in candidates:
            if os.path.isdir(candidate):
                match = re.search(r'checkpoint-(\d+)', os.path.basename(candidate))
                if match:
                    valid_checkpoints[int(match.group(1))] = candidate

        if not valid_checkpoints:
            print(f"Warning: No valid checkpoint directories found under {base_path}")
            return None

        latest_step = max(valid_checkpoints.keys())
        latest_checkpoint = valid_checkpoints[latest_step]
        print(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        print(f"Error finding latest checkpoint in {base_path}: {e}")
        # Fallback to simple max if regex fails or structure is unexpected
        try:
            latest = max(candidates, key=os.path.getmtime) # Fallback to modification time
            print(f"Falling back to latest modified: {latest}")
            return latest
        except ValueError:
            print(f"Could not determine latest checkpoint for {base_path}")
            return None


def load_all_inference_outputs(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads all .jsonl inference outputs from a folder."""
    data = {}
    jsonl_files = glob.glob(os.path.join(folder, "*_output.jsonl"))
    if not jsonl_files:
        print(f"Warning: No '*_output.jsonl' files found in {folder}")

    for file in jsonl_files:
        model_name = os.path.splitext(os.path.basename(file))[0].replace("_output", "")
        print(f"Loading inference output for model: {model_name} from {file}")
        try:
            data[model_name] = load_jsonl(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return data

def set_environment_variables(env_vars: Dict[str, str]):
    """Sets environment variables from a dictionary."""
    print("Setting environment variables...")
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = str(value)
            # print(f"Set {key}={value}") # Be cautious printing sensitive info
            print(f"Set {key}")
        else:
            print(f"Skipping setting {key} as value is None")

# Example of how to load from /etc/environment if absolutely needed,
# but prefer explicit config.
# def load_envs_from_file(file_path="/etc/environment"):
#     envs = {}
#     try:
#         with open(file_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line and not line.startswith('#') and '=' in line:
#                     key, value = line.split('=', 1)
#                     envs[key.strip()] = value.strip().replace('"', '')
#     except FileNotFoundError:
#         print(f"Warning: Environment file not found at {file_path}")
#     except Exception as e:
#         print(f"Error reading environment file {file_path}: {e}")
#     return envs