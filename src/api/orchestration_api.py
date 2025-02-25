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

from fastapi import FastAPI, HTTPException,  UploadFile, HTTPException, Form, File
from pydantic import BaseModel
from typing import Optional
import json
import requests

app = FastAPI()

EMBEDDING_SERVICE_URL = "http://localhost:5001/embed"
VECTORDB_SERVICE_URL = "http://localhost:5002/similarity-search/"
LLM_SERVICE_URL = "http://localhost:5003/generate-response/"

# Endpoint to run entire pipeline
@app.post("/orchestrate/")
def orchestrate_query(
    query: str = Form(...),
    type: str = Form(...),
    text_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    try:
        # Prepare payload for embedding service
        embedding_payload = {
            "type": type
        }
        # Handle files separately
        files = {}
        if text_file:
            files['text_file'] = (text_file.filename, text_file.file)
        if image_file:
            files['image_file'] = (image_file.filename, image_file.file)

        # Step 1: Send to the embedding service
        embedding_response = requests.post(
            EMBEDDING_SERVICE_URL,
            data=embedding_payload,
            files=files
        )

        if embedding_response.status_code != 200:
            raise HTTPException(status_code=embedding_response.status_code, detail=embedding_response.text)

        # Step 2: Extract the embedding from the response
        embedding_result = embedding_response.json()
        embedding = embedding_result["embedding"]

        # cleaned_results = []
        file_paths =[]
        text_search_payload = {
            "table": "kdb3",  # Replace with your actual table name
            "n": 1,  # Number of top results you want to retrieve
            "vectors": [embedding],  # The embedding you just got from the embedding service
            "distances": "cosine"  # Use 'cosine', 'euclidean', etc., depending on your distance metric
        }

        vectordb_response = requests.post(
            VECTORDB_SERVICE_URL,
            json=text_search_payload
        )

        if vectordb_response.status_code != 200:
            raise HTTPException(status_code=vectordb_response.status_code, detail=vectordb_response.text)

        results = vectordb_response.json()
        cleaned_results2 = []
        if "result" in results and "payload" in results["result"]:
            payload = results["result"]["payload"]
            # Assuming payload is a list of lists
            if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], list):
                for item in payload[0]:
                    cleaned_result = {
                        "cosine": item.get("cosine"),
                        "file_path": item.get("file_path"),
                        "media_type": item.get("media_type"),
                        "embeddings": item.get("embeddings")
                    }
                    cleaned_results2.append(cleaned_result)
                    file_paths.append(item.get("file_path"))

        print(file_paths)
        
        
        # Create the payload in the format expected by the LLMRequest in the LLM service
        payload = {
            "query": query,
            "similar_data": [{'file_path': 'images/'+image_file.filename}] +[{"file_path": file_path} for file_path in file_paths]
        }
        print(payload)

        try:
            # Make a POST request to the LLM service
            response = requests.post(LLM_SERVICE_URL, json=payload)

            # Check if the response status code is 200 (OK)
            if response.status_code == 200:
                # Parse and return the response from the LLM service
                return response.json()
            else:
                # Handle cases where the LLM service returns an error
                raise HTTPException(status_code=response.status_code, detail=response.text)
        
        except Exception as e:
            # Catch any exceptions and raise an HTTP error
            raise HTTPException(status_code=500, detail=f"Error contacting LLM service: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to test api call from orchestration service to embed service
@app.post("/test_embed/")
def test_embed():
    try:
        embedding_response = requests.post(
                EMBEDDING_SERVICE_URL
            )
        if embedding_response.status_code != 200:
                raise HTTPException(status_code=embedding_response.status_code, detail=embedding_response.text)

        # Handle response from embedding service
        embedding_result = embedding_response.json()

        # Return result or proceed with next steps
        return {"response": embedding_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))