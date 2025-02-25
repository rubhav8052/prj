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

from config.rag_config import (
    BLIP_GT_EMBEDDINGS_PATH,
    BLIP_MODEL_NAME,
    BLIP_PREPROCESS_METHOD,
    BLIP_TEST_EMBEDDINGS_PATH,
    GT_IMAGES_DIR,
    SEARCH_RESULTS_PATH,
    TEST_IMAGES_DIR,
    TOPK,
)
from model_management.evaluation.metrics.rag_metrics import Metrics
from embedding import BlipEmbeddingExtractor
from llm import gpt_4o, response_extractor
from simsearch import SimilaritySearch


def main() -> None:
    blip_extractor = BlipEmbeddingExtractor(preprocess_method=BLIP_PREPROCESS_METHOD, model_name=BLIP_MODEL_NAME)
    simsearch = SimilaritySearch(
    gt_embeddings_path=BLIP_GT_EMBEDDINGS_PATH, test_embeddings_path=BLIP_TEST_EMBEDDINGS_PATH, topk=TOPK
    )
    extractor = gpt_4o()

    blip_extractor(img_dir=GT_IMAGES_DIR, embeddings_path=BLIP_GT_EMBEDDINGS_PATH)
    blip_extractor(img_dir=TEST_IMAGES_DIR, embeddings_path=BLIP_TEST_EMBEDDINGS_PATH)

    simsearch.search()

    extractor(TEST_IMAGES_DIR, SEARCH_RESULTS_PATH)
    llm.generate_response(TEST_IMAGES_DIR, [], [])
    metrics_calculation = Metrics()
    results = metrics_calculation.calc_metrics(path=SEARCH_RESULTS_PATH)
    print(f"Color Accuracy: {results[0]}, Inlay Accuracy: {results[1]}")


if __name__ == "__main__":
    main()