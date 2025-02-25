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

import os

import pandas as pd
from cv2.typing import MatLike
from tqdm import tqdm


class EmbeddingExtractor:
    def __init__(self):
        pass

    def extract_embeddings(self, img: MatLike) -> MatLike:
        raise NotImplementedError

    def preprocess_image(self, path: str) -> MatLike:
        raise NotImplementedError

    def __call__(self, img_dir: str, embeddings_path: str) -> None:
        """Extract embeddings for given image directory.
        Args:
            img_dir (str): The path to the directory containing the images.
            embeddings_path (str): The path to save the extracted embeddings.
        Returns:
            None
        """
        embeddings_list = list()
        for filename in tqdm(os.listdir(img_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(img_dir, filename)
                img = self.preprocess_image(image_path)
                embeddings = self.extract_embeddings(img)
                embeddings_list.append({"filename": filename, "embeddings": embeddings})

        df = pd.DataFrame(embeddings_list)
        df.to_csv(embeddings_path, index=False)