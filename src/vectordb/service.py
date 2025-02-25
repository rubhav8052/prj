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

import aiohttp
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import requests
from fastapi import Request

load_dotenv()
KDBAI_ENDPOINT = os.getenv('KDBAI_ENDPOINT')
KDBAI_TOKEN = os.getenv('KDBAI_TOKEN')

# Function to insert data to KDB
def insert_data_to_kdb(payload: dict, HEADERS: dict):
    try:
        url = f"{KDBAI_ENDPOINT}/api/v1/insert"
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            return response.text
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to perform the search
def search_similar_data(payload: dict, HEADERS: dict):
    try:
        url = f"{KDBAI_ENDPOINT}/api/v1/search"
        response = requests.post(url, json=payload, headers=HEADERS)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

