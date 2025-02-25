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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from vectordb.models import LLMRequest, SimilarData
from llm.service import RAG_Setup
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import os

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise Exception("Google API Key is missing. Please set it in the environment variables.")
genai.configure(api_key=API_KEY)

# Initialize the FastAPI app
app = FastAPI()

# Use the Gemini model
vision_model = genai.GenerativeModel("gemini-1.5-flash")


# endpoint for generating responses with Gemini LLM using RAG
@app.post("/generate-response/")
async def generate_response(input_data: LLMRequest):
    try:
        # Prepare the RAG input with the query and similar data
        retrieved_data_for_RAG = RAG_Setup(input_data.similar_data, input_data.query)

        # Send data to Gemini Vision model for generation
        response = vision_model.generate_content(retrieved_data_for_RAG)
        return {"response": response.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endpoint for generating responses with Gemini LLM 
@app.post("/just_llm/")
async def just_llm( query: str):
    try:
        # Send data to Gemini Vision model for generation
        response = vision_model.generate_content(query)
        return {"response": response.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))