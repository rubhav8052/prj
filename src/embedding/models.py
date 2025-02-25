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

from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import Optional

class EmbedRequest(BaseModel):
    type: str = Form(...), 
    text: Optional[str] = Form(None),  
    text_file: Optional[UploadFile] = File(None),  
    image_file: Optional[UploadFile] = File(None)  
