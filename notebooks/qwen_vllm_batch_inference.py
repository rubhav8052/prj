import os
import json
import time
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from vehicle_prompt import vehicle_prompt
from qwen_vl_utils import process_vision_info
from tqdm import tqdm  # Import tqdm for progress tracking

# Initialize the model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
llm = LLM(
    #model="/home/rzr2kor/qwen-ft/qwen2_vl_7b-awq_4_epochs_3885_hist_eq", #Replace with path to model weights
    model_name,
    max_model_len=32768 if process_vision_info is None else 4096,
    max_num_seqs=5,
    limit_mm_per_prompt={"image": 16}, # max images per batch
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def load_qwen2_vl(image_path: str, question: str):
    """Prepare individual prompt and image data for each image."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant used for labelling Vehicle Images."},
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if process_vision_info is None:
        image_data = Image.open(image_path)
    else:
        image_data, _ = process_vision_info(messages)

    return prompt, image_data

def batch_inference(image_folder: str, output_file: str, batch_size: int = 16):
    """Perform batch inference with separate prompts for each image."""
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    time_list = []

    with open(output_file, "w") as f:
        f.write("[")  # Start JSON array
        first_entry = True

        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches", unit="batch"):
            batch_images = image_files[i:i + batch_size]
            prompts_and_images = [load_qwen2_vl(img, vehicle_prompt) for img in batch_images]
            prompts = [p[0] for p in prompts_and_images]
            images = [p[1] for p in prompts_and_images]

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
        avg_time_per_image = total_time / (len(image_files) - batch_size )
        print(f"Total inference time (excluding warmup): {total_time:.2f} seconds")
        print(f"Average inference time per batch: {avg_time_per_batch:.2f} seconds")
        print(f"Average inference time per image: {avg_time_per_image:.2f} seconds")

    print(f"Results saved to {output_file}")

# Define parameters
image_folder = "/home/rzr2kor/Vehicle_3885/Vehicle_3885" # Change path to image files
output_file = "results_batch_qwen.json" # Results file path 

# Run batch inference
batch_inference(image_folder, output_file)
