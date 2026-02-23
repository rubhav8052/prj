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
# from src.config.inference_config import DEEPSEEK_CONFIG

class DeepseekInference(BaseInference):
    def _initialize_llm(self) -> LLM:
        return LLM(
            model=self.config["model_name"],
            max_model_len=self.config["max_model_len"],
            max_num_seqs=self.config["max_num_seqs"],
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            dtype="float16",
            # ///////////////////////////////
            trust_remote_code=True # 🔑 Required for custom MoE architectures
        )

    def prepare_batch(self, image_files):
        images = [Image.open(img).convert("RGB") for img in image_files]
        prompt = self.config.get("prompt", "")
        prompts = [f"<|User|>: <image>\n {prompt} \n<|Assistant|>:" for _ in image_files]
        
        return [
            {"prompt": p, "multi_modal_data": {"image": [img]}} 
            for p, img in zip(prompts, images)
        ] 
# class DeepseekInference(BaseInference):
#     def _initialize_llm(self) -> LLM:
#         return LLM(
#             model=self.config["model_name"],
#             max_model_len=self.config["max_model_len"],
#             max_num_seqs=self.config["max_num_seqs"],
#             hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
#             dtype="float16"
#         )

#     def prepare_batch(self, image_files):
#         images = [Image.open(img).convert("RGB") for img in image_files]
#         prompt = self.config.get("prompt", "")
#         prompts = [f"<|User|>: <image>\n {prompt} \n<|Assistant|>:" for _ in image_files]
        
#         return [
#             {"prompt": p, "multi_modal_data": {"image": [img]}} 
#             for p, img in zip(prompts, images)
#         ] 