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

"""
Embedding extraction services.
Includes BLIP and other embedding extractors for image processing.
"""

from .base_extractor import EmbeddingExtractor
from .blip_extractor import BlipEmbeddingExtractor

__all__ = ['EmbeddingExtractor', 'BlipEmbeddingExtractor']