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


from PIL import Image
from vllm import LLM
from src.inference.local_lvms.base_inference import BaseInference
from src.config.inference_config import QWEN_CONFIG
from src.prompts.vehicle_prompt import vehicle_prompt
from qwen_vl_utils import process_vision_info

class QwenInference(BaseInference):
    def _initialize_llm(self) -> LLM:
        return LLM(
            model=self.config["model_name"],
            max_model_len=self.config["max_model_len"],
            max_num_seqs=self.config["max_num_seqs"],
            limit_mm_per_prompt={"image": 16},
            dtype="float16"
        )

    def prepare_batch(self, image_files):
        batch_data = []
        for img_path in image_files:
            messages = [
                {"role": "system", "content": "You are a helpful assistant used for labelling Vehicle Images."},
                {"role": "user", "content": [
                    {"type": "image", "image": img_path}, 
                    {"type": "text", "text": vehicle_prompt}
                ]}
            ]

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_data = (Image.open(img_path) if process_vision_info is None 
                         else process_vision_info(messages)[0])

            batch_data.append({
                "prompt": prompt,
                "multi_modal_data": {"image": [image_data]}
            })

        return batch_data 