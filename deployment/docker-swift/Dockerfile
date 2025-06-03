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

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

RUN conda update -n base -c conda-forge conda -y

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/mtl_onnx
# Create conda environment

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10.12 \
    pip=23.3.1 \
    nvidia/label/cuda-12.4.0::cuda-toolkit \
    git \
    -c anaconda -c pytorch -c nvidia -c conda-forge -y

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install tensorboard
RUN pip install requests==2.28.2 tqdm==4.65.0 rich==13.4.2
RUN pip install -U jupyterlab jupyter ipywidgets ipykernel
RUN pip install wheel opencv-python dynamic-yaml pandas scikit-image torch_kmeans
RUN pip install azureml-core
RUN pip install azure-storage-blob
RUN pip install onnx==1.17.0 onnxruntime-gpu==1.17 onnxsim==0.4.35
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install auto_gptq optimum -U
RUN pip install ms-swift deepspeed flash-attn pandas scikit-learn seaborn vllm==0.7.1 timm

RUN pip install qwen-vl-utils==0.0.10 transformers==4.48.2 -t qwen_libs/
RUN pip install transformers==4.41.0 peft==0.11.0 -t deepseek_libs/
RUN rm -rf /deepseek_libs/torch*
RUN rm -rf /deepseek_libs/triton*

                          
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH