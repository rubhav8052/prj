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

from setuptools import setup, find_packages

setup(
    name="auto-labeling-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated labeling system using LLM-RAG pipeline",
    keywords="labeling, llm, rag, machine-learning",
    python_requires=">=3.8",
)