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
Vector database services.
Includes similarity search and database management.
"""

from .similarity_search import SimilaritySearch
from .service import insert_data_to_kdb, search_similar_data

__all__ = ['SimilaritySearch', 'insert_data_to_kdb', 'search_similar_data']