# ============================================================================================================
# C O P Y R I G H T
# ------------------------------------------------------------------------------------------------------------
# \copyright (C) 2024 Robert Bosch GmbH and Cariad SE. All rights reserved.
# ============================================================================================================


FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230509.v1@sha256:21a5aff01e75574d94cd25802026a9ac5be1936dc2afe2973537f2c47abd4bc5

USER root

RUN conda update -n base -c conda-forge conda -y

# Create AzureML conda environment
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mtl_onnx

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10.12 \
    pip=23.3.1 \
    -c conda-forge -y

ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install Python dependencies
RUN pip install --no-cache-dir \
accelerate==1.4.0 \
torch==2.5.1 \
torchvision==0.20.1 \
transformers==4.48.2 \
vllm==0.7.1 \
numpy==1.26.4 \
pillow==10.4.0 \
qwen-vl-utils==0.0.10 \
pandas \
tqdm \
requests \
pydantic \
timm \
hydra-core==1.3.2 \
omegaconf==2.3.0

RUN pip install --no-cache-dir \
    azureml-mlflow==1.60.0 \
    mlflow==2.19.0 \
    setuptools==69.5.1


ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

