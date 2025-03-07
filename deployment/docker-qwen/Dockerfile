# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230509.v1@sha256:21a5aff01e75574d94cd25802026a9ac5be1936dc2afe2973537f2c47abd4bc5

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir accelerate==1.4.0  && \
    pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip3 install --no-cache-dir transformers==4.48.2 && \
    pip3 install --no-cache-dir vllm==0.7.1 && \
    pip3 install --no-cache-dir numpy==1.26.4 pillow==10.4.0 && \
    pip3 install --no-cache-dir qwen-vl-utils==0.0.10 && \
    pip3 install --no-cache-dir ray==2.42.0

COPY . .



