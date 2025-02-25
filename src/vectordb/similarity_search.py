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

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.rag_config import GT_METADATA_PATH, SEARCH_RESULTS_PATH, TEST_METADATA_PATH


class SimilaritySearch:
    def __init__(self, gt_embeddings_path: str, test_embeddings_path: str, topk: int = 5) -> None:
        self.gt_df = pd.read_csv(gt_embeddings_path)
        self.gt_embeddings = np.stack(
            self.gt_df["embeddings"].apply(lambda x: np.array(x[1:-1].split(",")).astype(float))
        )
        self.test_df = pd.read_csv(test_embeddings_path)
        self.test_df["embeddings"] = self.test_df["embeddings"].apply(
            lambda x: np.array(x[1:-1].split(",")).astype(float)
        )
        self.topk = topk

    def _get_topk_matches(self, test_embedding: np.ndarray) -> List[str]:
        """Search for the most similar images in the ground truth set.
        Args:
            test_embedding (np.ndarray): The embedding of the test image.
        Returns:
            List[str]: The filenames of the top k most similar images.
        """
        similarities = np.dot(self.gt_embeddings, test_embedding)
        most_similar_indices = np.argsort(similarities)[-self.topk :][::-1]
        return self.gt_df["filename"].iloc[most_similar_indices].tolist()

    def _get_attributes(self, metadata_path: str, filenames: List[str]) -> List[Dict[str, str]]:
        """Get the attributes of the images.
        Args:
            metadata_path (str): The path to the metadata file.
            filenames (List[str]): The list of filenames.
        Returns:
            List[Dict[str, str]]: The attributes of the images.
        """
        metadata_df = pd.read_csv(metadata_path).replace({np.nan: None})
        attributes = list()
        for filename in filenames:
            print(filename)
            attributes.append(metadata_df[metadata_df["file_name"] == filename].to_dict("records")[0])
        return attributes

    def search(self) -> None:
        """Search for the top k most similar images for each test image."""
        search_results = dict()
        for _, row in tqdm(self.test_df.iterrows()):
            filename, test_embedding = row["filename"], row["embeddings"]
            topk_matches = self._get_topk_matches(test_embedding)
            test_attributes = self._get_attributes(TEST_METADATA_PATH, [filename])
            topk_attributes = self._get_attributes(GT_METADATA_PATH, topk_matches)

            search_results[filename] = {
                "test_image": filename,
                "test_attributes": test_attributes,
                "topk_matches": {topk_matches[i]: topk_attributes[i] for i in range(self.topk)},
            }

        with open(SEARCH_RESULTS_PATH, "w") as json_file:
            json.dump(search_results, json_file, indent=4)