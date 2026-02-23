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

 
import os
import json
import time
from typing import List
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
# from src.config.inference_config import SAMPLING_PARAMS
 
class BaseInference:
    def __init__(self, model_config):
        self.config = model_config
        self.llm = self._initialize_llm()
        # 🔑 trust_remote_code=True is required for DeepSeek/Qwen config files
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_name"],
            trust_remote_code=True #//////////////////////////////
        )
 
    def _initialize_llm(self) -> LLM:
        raise NotImplementedError("Subclasses must implement _initialize_llm")
 
    def prepare_batch(self, image_files: List[str]):
        raise NotImplementedError("Subclasses must implement prepare_batch")
 
    def batch_inference(self, image_folder: str, output_file: str = None):
        """Perform batch inference."""
        output_file = output_file or self.config["output_file"]
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        image_files = sorted([
            os.path.join(image_folder, f) for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        time_list = []
 
        with open(output_file, "w") as f:
            f.write("[")
            first_entry = True
 
            for i in tqdm(range(0, len(image_files), self.config["batch_size"])):
                batch_images = image_files[i:i + self.config["batch_size"]]
                batch_data = self.prepare_batch(batch_images)
                # sampling_params = SamplingParams(**SAMPLING_PARAMS)
                sampling_params = SamplingParams(**self.config["sampling_params"])
 
                if i == 0:
                    outputs = self.llm.generate(batch_data, sampling_params=sampling_params)
                else:
                    start_time = time.time()
                    outputs = self.llm.generate(batch_data, sampling_params=sampling_params)
                    time_list.append(time.time() - start_time)
 
                for img_name, output in zip(batch_images, outputs):
                    result = {
                        "image_name": os.path.basename(img_name),
                        "output": output.outputs[0].text if output.outputs else ""
                    }
                    if not first_entry:
                        f.write(",\n")
                    json.dump(result, f, indent=4)
                    first_entry = False
 
            f.write("\n]")
 
        self._print_statistics(time_list, len(image_files))
        print(f"Results saved to {output_file}")
 
    def _print_statistics(self, time_list, total_images):
        if time_list:
            total_time = sum(time_list)
            avg_time_per_batch = total_time / len(time_list)
            avg_time_per_image = total_time / (total_images - self.config["batch_size"])
            print(f"Total inference time (excluding warmup): {total_time:.2f} seconds")
            print(f"Average inference time per batch: {avg_time_per_batch:.2f} seconds")
            print(f"Average inference time per image: {avg_time_per_image:.2f} seconds")