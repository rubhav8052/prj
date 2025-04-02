import os

# Common configurations
BASE_PATH = os.environ["AZUREML_DATAREFERENCE_VDEEPINGESTPROD"]
IMAGE_FOLDER = BASE_PATH + "/object_retrieval/AL_Data/Vehicle_3885"

# Model specific configurations
DEEPSEEK_CONFIG = {
    "model_name": "deepseek-ai/deepseek-vl2-tiny",
    "max_model_len": 4096,
    "max_num_seqs": 32,
    "batch_size": 5,
    "output_file": "outputs/results_batch_deepseek.json"
}

QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "model_path": BASE_PATH + "/object_retrieval/Qwen-2B weights/qwen2_vl_lora_sft_4_epochs_3885_128",
    "max_model_len": 4096,
    "max_num_seqs": 5,
    "batch_size": 16,
    "output_file": "outputs/results_batch_qwen.json"
}

# Common sampling parameters
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "max_tokens": 5000
} 