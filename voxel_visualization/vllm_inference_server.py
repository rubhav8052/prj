from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
import base64
import io

app = FastAPI()

# Load vLLM model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
llm = LLM(model=model_name, max_model_len=4096, max_num_seqs=1, limit_mm_per_prompt={"image": 1}, enforce_eager=True)

class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image
    prompt: str

def run_vllm(image_bytes, prompt):
    """Run inference with vLLM"""
    image = Image.open(io.BytesIO(image_bytes))
    messages = [
        {"role": "system", "content": "You are a helpful assistant used for labeling Vehicle Images."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]

    formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=300)
    outputs = llm.generate([{"prompt": formatted_prompt, "multi_modal_data": {"image": [image]}}], sampling_params=sampling_params)

    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

@app.post("/infer")
async def infer(request: InferenceRequest):
    image_bytes = base64.b64decode(request.image)
    response_text = run_vllm(image_bytes, request.prompt)
    return {"response": response_text}

# Run server: `uvicorn vllm_server:app --host 0.0.0.0 --port 5001 --workers 1`