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

"""Configuration file for the LLM-RAG pipeline."""

GT_IMAGES_DIR = "data/gt_images"
GT_METADATA_PATH = "data/gt_metadata.csv"

TEST_IMAGES_DIR = "data/test_images"
TEST_METADATA_PATH = "data/test_metadata.csv"

# Blip config
BLIP_PREPROCESS_METHOD = "blip"
BLIP_MODEL_NAME = "large"


# GPT Config
API_KEY = "your-api-key"
API_ENDPOINT = "https://gpt-pipeline.openai.azure.com/openai/deployments/gpt-4o-2024-05-13/chat/completions?api-version=2024-08-01-preview"
API_VERSION = "2024-08-01-preview"
DEPLOYMENT_NAME = "gpt-4o-2024-05-13"
FAILED_RESPONSE_PATH = "llm_results/gpt_failed.json"
RESULTS_PATH = "llm_results/gpt_results.json"
TOKEN_COUNT_FILE = "llm_results/gpt_token_count.json"


BLIP_GT_EMBEDDINGS_PATH = f"image_embeddings/gt_embeddings_{BLIP_PREPROCESS_METHOD}_blip.csv"
BLIP_TEST_EMBEDDINGS_PATH = f"image_embeddings/test_embeddings_{BLIP_PREPROCESS_METHOD}_blip.csv"

# Simsearch config
TOPK = 5
SEARCH_RESULTS_PATH = f"simsearch/search_results_{BLIP_PREPROCESS_METHOD}_top{TOPK}.json"