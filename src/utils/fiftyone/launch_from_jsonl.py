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
Given a jsonl file and all images in a specific folder, read it and launch a FiftyOne session.
"""

import json
import fiftyone as fo
from fiftyone import Sample
import os

def load_jsonl_to_fiftyone(jsonl_path:str, image_root:str, dataset_name:str="inlay_dataset"):
    dataset = fo.Dataset(dataset_name)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                image_rel_path = obj["images"][0].split("/")[-1]

                image_path= os.path.join(image_root, image_rel_path)

                label = None
                for convo in reversed(obj["conversations"]):
                    if convo["from"] == "assistant":
                        label = convo["value"]
                        break

                sample = fo.Sample(
                    filepath=image_path,
                    tags=[label]
                )

                if label:
                    sample["ground_truth"] = fo.Classification(label=label)

                dataset.add_sample(sample)

    print(f"Loaded {len(dataset)} samples into '{dataset_name}'")
    return dataset

image_root= "images"
jsonl_path = "tl_inlay_train_v1.jsonl"

dataset = load_jsonl_to_fiftyone(jsonl_path, image_root)
session = fo.launch_app(dataset)
session = session.wait()
