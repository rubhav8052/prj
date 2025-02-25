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

import streamlit as st
import requests

# Define the orchestration API URL
ORCHESTRATION_API_URL = "http://localhost:5004/orchestrate/"  # Replace with your FastAPI URL

# Streamlit UI
st.title("Orchestration Service UI")

# Step 1: Input the query
query = st.text_input("Enter your query", "")

# Step 2: Choose input type (either text or image)
input_type = st.selectbox("Select input type", ["text", "image"])

# Step 3: File uploader for text or image
uploaded_file = None
if input_type == "text":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
elif input_type == "image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

# Step 4: Submit button
if st.button("Submit Query"):
    if not query:
        st.error("Please enter a query.")
    elif not uploaded_file:
        st.error(f"Please upload a {input_type} file.")
    else:
        # Prepare the payload based on the input type and file
        files = {}
        if input_type == "text" and uploaded_file is not None:
            files['text_file'] = (uploaded_file.name, uploaded_file.getvalue())
        elif input_type == "image" and uploaded_file is not None:
            files['image_file'] = (uploaded_file.name, uploaded_file.getvalue())

        # Prepare the form data for the request
        data = {
            'query': query,
            'type': input_type
        }

        # Send the POST request to the orchestration API
        try:
            response = requests.post(ORCHESTRATION_API_URL, data=data, files=files)
            
            # Check the response
            if response.status_code == 200:
                # Display the response from the orchestration API
                st.success("Query successful!")
                st.json(response.json())
            else:
                st.error(f"Request failed with status code {response.status_code}")
                st.json(response.json())

        except Exception as e:
            st.error(f"Error contacting orchestration service: {str(e)}")