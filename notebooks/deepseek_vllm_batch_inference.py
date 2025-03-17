import os
import json
import time
from typing import List
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from vehicle_prompt import vehicle_prompt
from tqdm import tqdm 

model_name = "deepseek-ai/deepseek-vl2-tiny"
llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=32, #batch size 
              hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]})

processor = AutoProcessor.from_pretrained(model_name)

def batch_inference(image_folder: str, output_file: str, batch_size: int = 5):
    """Perform batch inference."""
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    time_list = []

    with open(output_file, "w") as f:
        f.write("[")  
        first_entry = True

        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches", unit="batch"):
            batch_images = image_files[i:i + batch_size]
            prompts = [f"<|User|>: <image>\n {vehicle_prompt} \n<|Assistant|>:" for _ in batch_images]
            images = [Image.open(img).convert("RGB") for img in batch_images]

            sampling_params = SamplingParams(
                temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=5000
            )
            if i == 0:
                outputs = llm.generate(
                [{"prompt": p, "multi_modal_data": {"image": [img]}} for p, img in zip(prompts, images)],
                sampling_params=sampling_params
            )
            else:
                start_time = time.time()
                outputs = llm.generate(
                    [{"prompt": p, "multi_modal_data": {"image": [img]}} for p, img in zip(prompts, images)],
                    sampling_params=sampling_params
                )
                end_time = time.time()
                time_list.append(end_time - start_time)

            for img_name, output in zip(batch_images, outputs):
                generated_text = output.outputs[0].text if output.outputs else ""
                result = {"image_name": os.path.basename(img_name), "output": generated_text}

                if not first_entry:
                    f.write(",\n")  
                json.dump(result, f, indent=4)
                first_entry = False

        f.write("\n]")  

    if time_list:
        total_time = sum(time_list)
        avg_time_per_batch = total_time / len(time_list)
        avg_time_per_image = total_time / (len(image_files)- batch_size)
        print(f"Total inference time (excluding warmup): {total_time:.2f} seconds")
        print(f"Average inference time per batch: {avg_time_per_batch:.2f} seconds")
        print(f"Average inference time per image: {avg_time_per_image:.2f} seconds")

    print(f"Results saved to {output_file}")

# Define parameters
image_folder = os.environ["AZUREML_DATAREFERENCE_VDEEPINGESTPROD"] + "/object_retrieval/AL_Data/Vehicle_3885" # Change path to image files
output_file = "results_batch_deepseek.json" # Results file path 

# Run batch inference
batch_inference(image_folder, output_file)
