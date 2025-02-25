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

import blip_inference as blip
import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from PIL import Image
from skimage.feature import hog

from embedding.base_extractor import EmbeddingExtractor


class BlipEmbeddingExtractor(EmbeddingExtractor):
    """Blip embedding extraction."""

    def __init__(self, preprocess_method: str = "hog", model_name: str = "large"):
        """
        Initializes an instance of the Blip class to extract embeddings.

        Args:
            img_dir (str): The image directory path to calculate embeddings.
            model_name (str): The name of the model to be used. Default is 'large'.

        Returns:
            None
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = blip.load(model_name, self.device)
        self.preprocessor = self._preprocessor_mapper(preprocess_method)

    def _preprocessor_mapper(self, method: str) -> callable:
        mapper = {
            "grayscale": self._img_to_grayscale,
            "hog": self._img_to_hog,
            "canny": self._img_to_canny,
            "canny_mask": self._img_color_overlay,
            "blip": self._img_to_blip,
        }
        try:
            return mapper[method]
        except KeyError:
            raise ValueError(f"Preprocess method '{method}' not found. Choose from {list(mapper.keys())}")

    def extract_embeddings(self, img: MatLike) -> MatLike:
        """
        Extracts image embeddings from the given image.

        Args:
            img (MatLike): The input image.

        Returns:
            MatLike: The extracted image embeddings.
        """
        image = Image.fromarray(img)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image).to(self.device)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0].tolist()

    def _img_to_canny(self, path: str) -> MatLike:
        """
        Converts an image to Canny edge detection.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The Canny edge detected image.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        otsu_threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(img, int(otsu_threshold), int(otsu_threshold * 1.5))
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_bgr

    def _img_color_overlay(self, path: str) -> MatLike:
        """
        Converts an image to color overlay.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The color overlay image.
        """
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(gray, int(otsu_threshold), int(otsu_threshold * 1.5))
        mask = np.zeros_like(img)
        mask[edges != 0] = [255, 255, 255]

        overlay = cv2.addWeighted(img, 1, mask, 1, 0)
        return overlay

    def _img_to_grayscale(self, path: str) -> MatLike:
        """
        Converts an image to grayscale.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The grayscale image.
        """
        img = cv2.imread(path)
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        pil_image = Image.fromarray(image_bgr)
        return pil_image

    def _img_to_hog(self, path: str) -> MatLike:
        """
        Converts an image to HOG.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The HOG image.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, hog_img = hog(img, orientations=8, pixels_per_cell=(1, 1), cells_per_block=(1, 1), visualize=True)
        bgr_image = cv2.merge([hog_img, hog_img, hog_img])
        return bgr_image

    def _img_to_blip(self, path: str) -> MatLike:
        """
        Passes image directly to Blip without preprocessing.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The BGR image.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def preprocess_image(self, path: str) -> MatLike:
        """
        Preprocesses the given image.

        Args:
            path (str): The path to the image file.

        Returns:
            MatLike: The preprocessed image.
        """
        return self.preprocessor(path)