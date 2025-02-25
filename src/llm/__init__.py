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
Language model services.
Includes GPT and response extraction functionality.
"""

from .gpt import gpt_4o
from .response_extractor import ResponseExtractor

__all__ = ['gpt_4o', 'ResponseExtractor']