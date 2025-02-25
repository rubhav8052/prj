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

Welcome to the **LLM RAG Pipeline repository!** This project facilitates efficient embeddings extraction, similarity search on generated embeddings, and inference using a Multimodal Large Language Model (LLM).

The pipeline is designed for seamless integration of Retrieval-Augmented Generation (RAG) workflows with LLMs to enhance knowledge retrieval and multimodal understanding.

# Features
- **Embeddings Extraction:** Convert text, images, or other modalities into embeddings for retrieval and search tasks.
- **Similarity Search:** Perform similarity-based searches to find relevant content using cosine similarity or other distance metrics.
- **Inference with Multimodal LLM:** Enable inference tasks such as question answering, summarization, or other generation tasks by leveraging a multimodal LLM.
- **Efficient Data Handling:** Designed to handle large datasets with speed and scalability.

## Installation

To install the project, follow these steps:

### Step 1: Clone the repository
First, clone the repository:

```bash
git clone https://github.com/mgx2kor/llm-rag-pipeline.git
cd llm-rag-pipeline

```
### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Data Configuration

To configure the paths for your data, you can modify the **file paths** in the `config.py` file. The paths define where the ground truth (GT) images, test images, and their respective metadata are stored.

#### File Paths in `config.py`:

The following file paths are specified in the `config.py` file:

1. **Ground Truth (GT) Images for BLIP:**
   - Path: `data/gt_images`
   - Description: This directory contains the ground truth images that will be used by the BLIP model for processing.

2. **Ground Truth Metadata for BLIP:**
   - Path: `data/gt_metadata.csv`
   - Description: This CSV file contains metadata related to the ground truth images, such as labels, tags, or any other necessary information for processing.

3. **Test Images for BLIP:**
   - Path: `data/test_images`
   - Description: This directory contains the test images that will be processed by the BLIP model.

4. **Test Metadata for BLIP:**
   - Path: `data/test_metadata.csv`
   - Description: This CSV file contains metadata related to the test images, providing additional context or labels that may be needed for processing.
  
### Step 4 : GPT Configuration

To configure the LLM, you can modify the GPT Config parameters in the `config.py` file.

#### Configuration Parameters:

1. **API_KEY**:
   - Description: This is your unique API key used to authenticate with the GPT API.

2. **API_ENDPOINT**:
   - Description: This is the endpoint URL used to access the GPT API. The URL is generally provided by OpenAI (or another provider if you are using a custom GPT deployment).

3. **API_VERSION**:
   - Description: This specifies the version of the GPT API you are using.

4. **DEPLOYMENT_NAME**:
   - Description: The name of the GPT model deployment.

5. **FAILED_RESPONSE_PATH**:
   - Description: This file path specifies where failed responses from GPT should be saved. 

6. **RESULTS_PATH**:
   - Description: This file path specifies where the successful results from GPT will be stored. 

7. **TOKEN_COUNT_FILE**:
   - Description: This file path specifies where the token count of the run with GPT will be stored.
  
### Step 5: Run the Workflow

To trigger the flow of **BLIP**, **RAG**, and **LLM**, run the `main.py` script. This will execute the entire pipeline and handle the process of **extracting embeddings, performing similarity search, generating responses with the LLM and calculate metrics** for Color and Inlay of Traffic Light Bulb.

Run the following command in your terminal:

```bash
python main.py

