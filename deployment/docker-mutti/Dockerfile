# ============================================================================================================
# C O P Y R I G H T
# ------------------------------------------------------------------------------------------------------------
# \copyright (C) 2025 Robert Bosch GmbH and Cariad SE. All rights reserved.
# ============================================================================================================
FROM rayproject/ray:2.43.0-py310-cpu
# Since modern deep learning libaries come with their own cuda, cudnn, etc., we avoid using the nvidia cuda image
# and reduce docker image size. If you still prefer to use a nvidia cuda image, you can use the one below:
# FROM rayproject/ray:2.43.0-py310-cu125

# This Dockerfile is based on the official Ray Docker image with user and miniconda setup.
# For details check the Ray dockerfiles: https://github.com/ray-project/ray/blob/master/docker/base-deps/Dockerfile

# defaults from ray base image (do not change)
ARG USERNAME=ray
ARG GROUP=users
ARG UID=1000
ARG GID=100

# Switch to root to install system dependencies
USER root

### Install some custom software that is missing from the regular container
RUN apt-get update && apt-get install -y --no-install-recommends \
    tree \
    bash-completion \
    file \
    curl \
    # required by opencv (cv2)
    libgl1 \
    libglib2.0-0 \
    libtbb-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    openssh-server \
    build-essential

# Install CUDA 12.5 and cuDNN 9.3 from the NVIDIA repository (compatible with tf 2.19.0)
# cuda-libraries-12-5 contains the runtime libraries for CUDA 12.5 (in contrast to cuda-toolkit-12-5)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12=9.3.0.75-1 libcudnn9-dev-cuda-12=9.3.0.75-1 cuda-libraries-12-5

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Switch to standard user to install conda and python packages
USER $UID
# Install inference libraries
RUN pip --no-cache-dir install --upgrade pip ipython setuptools packaging && \
    pip --no-cache-dir install nvidia-cuda-nvcc-cu12==12.5.82 tensorflow==2.19.0 && \
    # pip --no-cache-dir install tensorflow[and-cuda]==2.19.0 && \
    pip --no-cache-dir install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip --no-cache-dir install onnxruntime-gpu==1.21.0 && \
    pip --no-cache-dir install accelerate transformers vllm==0.7.2 && \
    pip --no-cache-dir install azure-cli-core==2.64.0 azureml-core && \
    pip --no-cache-dir install timm==1.0.15 qwen-vl-utils==0.0.10


# DOCKER-LAYER-CACHE works only for the instructions before copying the repo code.
# ============================================================================================================

# Switch to root to copy repo code and models
USER root
# Create a working directory, mkdir instruction is needed so our USER becomes the owner.
# Even though WORKDIR would create the non-existing directory, its owner would be ROOT.
ENV REPO=/home/$USERNAME/repo
RUN mkdir -p $REPO /outputs
WORKDIR $REPO

# Copy repo code to image and update permissions
# COPY --chown=$UID:$GID . $REPO/
# RUN ln -s $REPO/models /models && \
#     ln -s $REPO/config /config && \
#     chown $UID:$GID $REPO/ /config /models /outputs

# Switch to standard user (uid 1000) which is also the user expected by our Argo dataloop.
USER $UID

# Make cudnn and cuda libraries discoverable by tensorflow and onnxruntime
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# # Install repo and requirements using ADA artificatory
# RUN --mount=type=secret,id=PIP_INDEX_URL,uid=1000 \
#     --mount=type=secret,id=PIP_EXTRA_INDEX_URL,uid=1000 \
#     export PIP_INDEX_URL=$(cat /run/secrets/PIP_INDEX_URL) && \
#     export PIP_EXTRA_INDEX_URL=$(cat /run/secrets/PIP_EXTRA_INDEX_URL) && \
#     pip --no-cache-dir install .[test] && \
#     pip --no-cache-dir install --no-deps bosch-openlabel==2.5.2
