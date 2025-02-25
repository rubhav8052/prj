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

from vectordb.models import LLMRequest, SimilarData
from typing import List
from PIL import Image
import os


def RAG_Setup(similar_data: List[SimilarData], query: str):
    retrieved_data_for_RAG = [f"You will answer the given prompt using attached content: {query}"]

    for data in similar_data:
        # Extract file path from the SimilarData object
        file_path = data['file_path'] if isinstance(data, dict) else data.file_path

        # Determine if the file is an image or text by the file extension
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Handle image files
            image = Image.open(file_path)
            retrieved_data_for_RAG.append(image)
        elif file_path.lower().endswith('.txt'):
            # Handle text files
            with open(file_path, 'r') as file:
                text = file.read()
            retrieved_data_for_RAG.append(text)

    return retrieved_data_for_RAG