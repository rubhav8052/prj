## Setup and Installation

1. Install requirements for voxel : `pip install -r requirements.txt`
2. Configure paths 

\
Requirements to run inference using vllm : deployment/docker-qwen/requirements.txt 

FiftyOne recognizes plugins by searching for fiftyone.yml or fiftyone.yaml files within your plugins directory.

Below is an example of a plugin directory with a typical Python plugin :

- '/path/to/your/plugins/dir/'
   - my-py-plugin/
    -    fiftyone.yml
    -   __init__.py
    -   requirements.txt

Below is the directory structure of our python voxel plugin:

- /voxel_visualization/
    - voxel_plugin/
        - fiftyone.yml
        - __init__.py
        - requirements.txt



vllm_inference_server.py :

This FastAPI application provides an endpoint for running inference on images using the Qwen2-VL-2B-Instruct model with vLLM. The API accepts a base64-encoded image along with a text prompt and returns the generated response.

## Usage

Start the inference server by running :
Run server: `uvicorn vllm_inference_server:app --host 0.0.0.0 --port 5001 --workers 1`

Run voxel_visualization.ipynb notebook 