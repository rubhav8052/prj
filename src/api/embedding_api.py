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

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
import os
import io
from typing import Optional
from embedding.service import data_to_embedding, save_embedding_to_json, generate_timestamped_filename

# Define directories for storing files
TEXT_DIR = "text/"
IMAGE_DIR = "images/"
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI()

@app.post("/embed")
async def embed(
    type: str = Form(...),  # Form input for type (text or image)
    text: Optional[str] = Form(None),  # Form input for text (optional)
    text_file: Optional[UploadFile] = File(None),  # File input for text file (optional)
    image_file: Optional[UploadFile] = File(None)  # File input for image file (optional)
):
    if type not in ['text', 'image']:
        raise HTTPException(status_code=400, detail="Unsupported type")

    try:
        # Handle image input, must be a file
        if type == 'image':
            if image_file:
                file_path = os.path.join(IMAGE_DIR, image_file.filename)
                file_data = await image_file.read()
                image = Image.open(io.BytesIO(file_data))
                image.save(file_path, format="JPEG")  # Save the image in JPEG format
                # Pass the image path to the embedding function
                embedding = data_to_embedding(file_path, 'image')
                save_embedding_to_json(file_path, type, embedding)
            else:
                raise HTTPException(status_code=400, detail="Image file must be provided for image type")
        
        # Handle text input, either as direct text or from a file
        elif type == 'text':
            if text:
                file_path = generate_timestamped_filename(TEXT_DIR, "txt")
                with open(file_path, "w") as f:
                    f.write(text)
                embedding = data_to_embedding(file_path, 'text')
                save_embedding_to_json(file_path, type, embedding)
            elif text_file:
                file_path = os.path.join(TEXT_DIR, text_file.filename)
                with open(file_path, "wb") as f:
                    f.write(await text_file.read())
                embedding = data_to_embedding(file_path, 'text')
                save_embedding_to_json(file_path, type, embedding)
            else:
                raise HTTPException(status_code=400, detail="Either text or text file must be provided for text type")

        return {"embedding": embedding.tolist()}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/test")
async def test():
    return {"embedding": "test successful"}