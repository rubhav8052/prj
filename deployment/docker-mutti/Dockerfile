# ============================================================================================================
# C O P Y R I G H T
# ------------------------------------------------------------------------------------------------------------
# \copyright (C) 2024 Robert Bosch GmbH and Cariad SE. All rights reserved.
# ============================================================================================================
FROM rayproject/ray:2.31.0-py310-cpu
# Since modern deep learning libaries come with their own cuda, cudnn, etc., we avoid using the nvidia cuda image
# and reduce docker image size. If you still prefer to use a nvidia cuda image, you can use the one below:
# FROM rayproject/ray:2.31.0-py310-cu118  (this is based on nvidia cuda image)

# This Dockerfile is based on the official Ray Docker image with user and miniconda setup.
# For details check the Ray dockerfiles: https://github.com/ray-project/ray/blob/master/docker/base-deps/Dockerfile

ARG USERNAME=ray
ARG GROUP=users
ARG UID=1000
ARG GID=100
ARG TENANT_ID=a6c60f0f-76aa-4f80-8dba-092771d439f0

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
    openssh-server

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb && \
    dpkg -i libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb && \
    rm libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb
    

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Switch to standard user to install conda and python packages
USER $UID
# Install cudatoolkit with conda (for onnxruntime-gpu support) and cudnn, torch, tensorflow with pip
RUN conda install -y cudatoolkit=11.8 && conda clean --all --yes
RUN pip --no-cache-dir install --upgrade pip ipython setuptools packaging && \
    pip --no-cache-dir install tensorflow[and-cuda]==2.14.* && \
    pip --no-cache-dir install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip --no-cache-dir install azure-cli-core==2.58.0 azureml-core && \
    pip --no-cache-dir install accelerate==1.4.0 transformers==4.48.2 qwen-vl-utils==0.0.10 && \
    pip --no-cache-dir install pillow==10.4.0 jupyter notebook && \
    pip --no-cache-dir install https://github.com/vllm-project/vllm/releases/download/v0.7.3/vllm-0.7.3+cu118-cp38-abi3-manylinux1_x86_64.whl && \
    pip --no-cache-dir install -U --force-reinstall xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118 && \
    pip --no-cache-dir install numpy==1.26.4

# DOCKER-LAYER-CACHE works only for the instructions before copying the repo code.
# ============================================================================================================

# Switch to root to copy repo code and models
USER root

# Create a working directory, mkdir instruction is needed so our USER becomes the owner.
# Even though WORKDIR would create the non-existing directory, its owner would be ROOT.
# ENV REPO /home/$USERNAME/repo
# RUN mkdir -p $REPO /outputs
# WORKDIR $REPO

# # Copy repo code to image and update permissions
# COPY --chown=$UID:$GID . $REPO/
# RUN ln -s $REPO/models /models && \
#     ln -s $REPO/config /config && \
#     chown $UID:$GID $REPO/ /config /models /outputs

# # Switch to standard user to install repo and requirements
# USER $UID
# # Install repo and requirements using ADA artificatory
# RUN --mount=type=secret,id=PIP_INDEX_URL,uid=1000 \
#     --mount=type=secret,id=PIP_EXTRA_INDEX_URL,uid=1000 \
#     export PIP_INDEX_URL=$(cat /run/secrets/PIP_INDEX_URL) && \
#     export PIP_EXTRA_INDEX_URL=$(cat /run/secrets/PIP_EXTRA_INDEX_URL) && \
#     pip --no-cache-dir install .[test]

# # Switch to root to link libraries
# USER root
# # Make cuda/cudnn libraries permanently and system-wide available using ldconfig (instead of exporting to LD_LIBRARY_PATH).
# # Note: We use double-quotes to ensure that $CONDA_PREFIX & $CUDNN_PATH will expand NOW (build time).
RUN mkdir -p /etc/ld.so.conf.d && \
    export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")) && \
    echo "${CONDA_PREFIX}/lib" >  /etc/ld.so.conf.d/cuda.conf && \
    echo "${CUDNN_PATH}/lib"   >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

# # Switch to standard user (uid 1000) which is also the user expected by our Argo dataloop.
# USER $UID