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

from fastapi import FastAPI, HTTPException, Request, Body, File, UploadFile
import requests
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from vectordb.models import SearchData 
from vectordb.service import insert_data_to_kdb, search_similar_data
import pandas as pd
from typing import List
import numpy as np
import aiohttp

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

KDBAI_ENDPOINT = os.getenv('KDBAI_ENDPOINT')
KDBAI_TOKEN = os.getenv('KDBAI_TOKEN')
HEADERS = {
    "Content-Type": "application/json",
    "X-Api-Key": KDBAI_TOKEN
}

# Endpoint to create the table
@app.post("/create-table")
def create_table():
    with open('table_schema.json', 'r') as schema_file:
        table_schema = schema_file.read()

    try:
        response = requests.post(
            f"{KDBAI_ENDPOINT}/api/v1/config/table/kdb3",
            headers=HEADERS,
            data=table_schema
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return {"message": "Table created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to upload and insert a .json file
@app.post("/insert-data-file/")
def insert_data_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded JSON file
        file_content = file.file.read()
        data = json.loads(file_content)

        # Ensure the file contains the expected structure
        if "table" not in data or "rows" not in data:
            raise HTTPException(status_code=400, detail="Invalid JSON structure. 'table' and 'rows' keys are required.")

        # Insert the data into KDB
        result = insert_data_to_kdb(data,HEADERS)
        return {"result": result}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="The uploaded file is not valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to run similarity search
@app.post("/similarity-search/")
def similarity_search(search_data: SearchData):
    try:
        # Prepare the payload for the request
        payload = {
            "table": search_data.table,
            "n": search_data.n,
            "vectors": search_data.vectors,
            "distances": search_data.distances
        }

        # Perform the search
        result = search_similar_data(payload, HEADERS)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ENdpoint to query all data in the table
@app.post("/query-all")
def query_all_data():
    try:
        payload = {
            "table": "kdb3"
        }
        response = requests.post(
            f"{KDBAI_ENDPOINT}/api/v1/data",
            json=payload,
            headers=HEADERS
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
