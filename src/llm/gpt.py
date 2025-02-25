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

import base64
import json
import os
import time

from cv2.typing import MatLike
from openai import AzureOpenAI

from config.rag_config import API_ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT_NAME
from language_models.prompt import prompt
from llm.response_extractor import ResponseExtractor


class gpt_4o(ResponseExtractor):
    def __init__(self):
        super().__init__

    def encode_image(self, image_path: str) -> MatLike:
        """
        Encode an image file as a base64 string.

        This method reads an image from the specified file path, encodes its content in
        base64 format, and returns the resulting string. This can be useful for
        transmitting images over text-based protocols or embedding images in formats
        such as JSON.

        Args:
            image_path (str): The path to the image file that needs to be encoded.

        Returns:
            MatLike: A base64-encoded string representation of the image.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If there is an error reading the image file.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image Not found")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_response(self, image_path: str, color_list: list[str], inlay_list: list[str]) -> dict:
        """This method utilizes Azure's OpenAI service, gpt-4o model to analyze the traffic light image
        and extract relevant attributes. The extracted attributes are formatted as a JSON object
        containing the color and inlay shape of the traffic light bulb.

        Args:
        - image_path (str): The file path of the image to be analyzed.

        Returns:
        dict: A dictionary containing the response from the language model
        """
        client = AzureOpenAI(azure_endpoint=API_ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

        base64image = self.encode_image(image_path)
        formatted_prompt = prompt.format(color_list=color_list, inlay_list=inlay_list)
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant trained to label traffic light images. Your task is to analyze images of traffic lights and provide detailed information about the color and inlay of the bulbs in the traffic lights. You should fetch the attributes in the prompt in the format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64image}"}},
                        {"type": "text", "text": formatted_prompt},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        # generated_text = response.choices[0].message.content

        return response

    def rag(self, img: str, search_results_path: str) -> list[str]:
        rag_list = []
        d = {}
        color_list, inlay_list = [], []
        with open(search_results_path, "r") as f:
            data = json.load(f)
        for file, val in data.items():
            if file == img:
                for _, value in val["topk_matches"].items():
                    color_list.append(value["color"])
                    inlay_list.append(value["inlay"])
                d[file] = {"color": color_list, "inlay": inlay_list}
                rag_list.append(d)
        return rag_list