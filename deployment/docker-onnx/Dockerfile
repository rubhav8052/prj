FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230509.v1@sha256:21a5aff01e75574d94cd25802026a9ac5be1936dc2afe2973537f2c47abd4bc5

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y

RUN conda update -n base -c conda-forge conda -y

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/mtl_onnx
# Create conda environment

RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10.12 \
    pip=23.3.1 \
    pytorch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    pytorch-cuda=11.8 \
    -c anaconda -c pytorch -c nvidia -c conda-forge -y

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install requests==2.28.2 tqdm==4.65.0 rich==13.4.2
RUN pip install -U jupyterlab jupyter ipywidgets ipykernel
RUN pip install wheel opencv-python dynamic-yaml pandas scikit-image torch_kmeans
RUN pip install azureml-core
RUN pip install azure-storage-blob
RUN pip install onnx==1.17.0 onnxruntime-gpu==1.17 onnxsim==0.4.35 transformers==4.38.1
                          
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH