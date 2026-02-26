# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================









FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230509.v1@sha256:21a5aff01e75574d94cd25802026a9ac5be1936dc2afe2973537f2c47abd4bc5

# -------------------------------------------------
# System dependencies (merged, CUDA-safe)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libtbb-dev \
    git \
    build-essential \
    tree \
    bash-completion \
    file \
    curl \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Update conda
# -------------------------------------------------
RUN conda update -n base -c conda-forge conda -y

# -------------------------------------------------
# Create AzureML conda environment
# -------------------------------------------------
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mtl_onnx

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10.12 \
    pip=23.3.1 \
    -c conda-forge -y

ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH



# -------------------------------------------------
# Core Python utilities
# -------------------------------------------------
RUN pip install --no-cache-dir \
    wheel \
    numpy==1.26.4 \
    requests==2.28.2 \
    tqdm==4.65.0 \
    rich==13.4.2 \
    pandas \
    scikit-image \
    scikit-learn==1.7.0 \
    seaborn \
    dynamic-yaml \
    opencv-python \
    torch_kmeans \
    click==8.1.7 \
    python-dotenv==1.0.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    pyyaml==6.0.1 \
    tensorboard \
    jupyterlab \
    jupyter \
    ipywidgets \
    ipykernel \
    pillow==10.4.0

# -------------------------------------------------
# AzureML SDK (pinned & stable)
# -------------------------------------------------
RUN pip install --no-cache-dir \
    azureml-core==1.60.0.post1 \
    azure-storage-blob
    
# ///////////////////////////////////////////////
RUN pip install --no-cache-dir \
    azureml-mlflow==1.60.0 \
    mlflow==2.19.0 \
    setuptools==69.5.1
# ///////////////////////////////////////////////


# -------------------------------------------------
# ONNX stack
# -------------------------------------------------
RUN pip install --no-cache-dir \
    onnx==1.17.0 \
    onnxruntime-gpu==1.17.0 \
    onnxsim==0.4.35




# -------------------------------------------------
# PyTorch CUDA runtime (latest safe set)
# -------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

    
# -------------------------------------------------
# LLM / VLM stack (fully compatible)
# -------------------------------------------------
# Core HF stack
# transformers==4.48.2 \
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    transformers==4.46.3 \
    accelerate==1.4.0 \
    peft==0.11.0 \
    ms-swift==3.6.1 \
    timm==1.0.15 \
    qwen-vl-utils==0.0.10 \
    optimum==1.17.1

# Heavy GPU wheels (separate)
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    vllm==0.6.6 \
    auto_gptq
    

# -------------------------------------------------
# Runtime hygiene
# -------------------------------------------------
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

