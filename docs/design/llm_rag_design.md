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

# LLM-RAG

To Build all containers and start the services run the following command:

bash startup.sh

Note: 
If you face connection timed out error, rerun the above command ("May happen due to proxy issues")

Once all containers are running, you can access each service via swagger at the endpoint http://localhost:PORT/docs on your browser

Ports for the different services are :
Embedding service : 5001
VectorDB service : 5002
LLM service : 5003
Orchestration service :5004

To access the UI hosted by Streamlit :
http://localhost:8501