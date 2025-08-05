import argparse
import base64
import io

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import uvicorn

# -----------------------------
# Parse runtime arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["qwen", "deepseek"], default="qwen",
                    help="Choose which model to load at startup.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=5001, help="Server port")
parser.add_argument("--workers", type=int, default=1, help="Number of workers for Uvicorn")
args = parser.parse_args()

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI()
model_type = args.model

# Globals for model and processor
llm = None
processor = None

# -----------------------------
# Load model on startup (only once per worker)
# -----------------------------
@app.on_event("startup")
async def load_model():
    global llm, processor

    if llm is not None:
        return  # Already loaded

    if model_type == "qwen":
        MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=4096,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            dtype="float16"
        )

    elif model_type == "deepseek":
        MODEL_NAME = "deepseek-ai/deepseek-vl2-tiny"
        processor = None
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=4096,
            max_num_seqs=1,
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            dtype="float16",
            enforce_eager=True
        )

# -----------------------------
# Request model
# -----------------------------
class InferenceRequest(BaseModel):
    image: str  # Base64-encoded image
    prompt: str # Text prompt

# -----------------------------
# Inference functions
# -----------------------------
def run_qwen(image_bytes, prompt):
    image = Image.open(io.BytesIO(image_bytes))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]
    formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=300)
    outputs = llm.generate(
        [{"prompt": formatted_prompt, "multi_modal_data": {"image": [image]}}],
        sampling_params=sampling_params
    )
    num_input_tokens = len(outputs[0].prompt_token_ids)
    num_output_tokens = len(outputs[0].outputs[0].token_ids)
    inference_time = outputs[0].metrics.finished_time - outputs[0].metrics.arrival_time
    print(f"Input tokens: {num_input_tokens}, Output tokens: {num_output_tokens}, Inference time: {inference_time:.2f} seconds")
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

def run_deepseek(image_bytes, prompt):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    formatted_prompt = f"<|User|>: <image>\n{prompt}\n<|Assistant|>:"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=300)
    outputs = llm.generate(
        [{"prompt": formatted_prompt, "multi_modal_data": {"image": [image]}}],
        sampling_params=sampling_params
    )
    num_input_tokens = len(outputs[0].prompt_token_ids)
    num_output_tokens = len(outputs[0].outputs[0].token_ids)
    inference_time = outputs[0].metrics.finished_time - outputs[0].metrics.arrival_time
    print(f"Input tokens: {num_input_tokens}, Output tokens: {num_output_tokens}, Inference time: {inference_time:.2f} seconds")
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.post("/infer")
async def infer(request: InferenceRequest):
    try:
        image_bytes = base64.b64decode(request.image)

        if model_type == "deepseek":
            response_text = run_deepseek(image_bytes, request.prompt)
        else:
            response_text = run_qwen(image_bytes, request.prompt)

        return {"response": response_text}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# -----------------------------
# Python launch entrypoint
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "vllm_inference_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers
    )
#Usage python3 vllm_inference_server.py --model qwen/deepseek