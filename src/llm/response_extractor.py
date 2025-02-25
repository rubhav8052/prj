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

import json
import os
import time

import openai
import pandas as pd
from cv2.typing import MatLike
from tqdm import tqdm

from config.rag_config import FAILED_RESPONSE_PATH, RESULTS_PATH, TOKEN_COUNT_FILE


class ResponseExtractor:
    def __init__(self):
        pass

    def encode_image(self, image_path: str) -> MatLike:
        raise FileNotFoundError

    def generate_response(self, image_path: str, color_list: list[str], inlay_list: list[str]) -> dict:
        raise NotImplementedError

    def rag(self, img: str, search_results_path: str) -> list[str]:
        raise FileNotFoundError

    def __call__(self, image_dir: str, search_results_path: str) -> dict:
        gpt_response_failed = {}
        gpt_response = {}
        gpt_token_count = {}
        max_retries_response = 5
        for filename in tqdm(os.listdir(image_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(image_dir, filename)
                rag_list = self.rag(filename, search_results_path)


            response_output = None
            max_retries = 3
            retries = 0

            while retries < max_retries:
                try:
                    for ele in rag_list:
                        for _, val in ele.items():
                            color_list = val['color']
                            inlay_list = val['inlay']
                    response_output = self.generate_response(image_path, color_list, inlay_list)

                    if response_output and response_output.choices[0].message.content is not None:
                        break
                    print(f"Response is None on attempt {retries + 1}. Retrying...")
                    retries += 1
                    time.sleep(2)

                except openai.BadRequestError as err:
                    if "Invalid content type. image_url is only supported by certain models." in str(err):
                        print(f"BadRequestError on attempt {
                            retries + 1}: {err}")
                        retries += 1
                        if retries == max_retries:
                            print(f"Failed to get a valid response for image after {
                                gretries} retries: {filename}")

                            gpt_response_failed[filename] = {
                                "test_image": filename,
                                "response": err
                            }
                            with open(FAILED_RESPONSE_PATH, "a") as f:
                                json.dump(gpt_response_failed, f, indent=4)
                        print("Retrying...")
                        time.sleep(2)
                        continue

                except ValueError as ve:
                    print(f"ValueError on attempt {retries + 1}: {ve}")
                    retries += 1
                    if retries == max_retries:
                        print(f"Failed to get a valid response for image after {
                            retries} retries: {filename}")
                        gpt_response_failed[filename] = {
                            "test_image": filename,
                            "response": ve
                        }
                        with open(FAILED_RESPONSE_PATH, "a") as f:
                            json.dump(gpt_response_failed, f, indent=4)

                    print("Retrying...")

            if response_output is None or response_output.choices[0].message.content is None:
                retries = 3
                while retries < max_retries_response:
                    print(f"Attempting additional retries for image: {
                        filename} (Attempt {retries + 1})")
                    time.sleep(1)
                    response_output = self.generate_response(image_path, color_list, inlay_list)

                    if response_output and response_output.choices[0].message.content is not None:
                        break

                    print(f"Response is still None on additional attempt {
                        retries + 1}. Retrying...")
                    retries += 1
                    time.sleep(2)

            if response_output is None or response_output.choices[0].message.content is None:
                print(f"No valid response for image after all retries: {
                    filename}. Logging to failures.")
                gpt_response_failed[filename] = {
                    "test_image": filename,
                    "response": "No valid response"
                }
                with open(FAILED_RESPONSE_PATH, "w") as f:
                    json.dump(gpt_response_failed, f, indent=4)
            else:
                if '\n' in response_output.choices[0].message.content:
                    response_output.choices[0].message.content = "".join(
                        response_output.choices[0].message.content.split()
                    )
                gpt_response[filename] = {
                    "test_image": filename,
                    "response": str(response_output.choices[0].message.content)
                }
                gpt_token_count[filename] = {
                    "Token Count": str(response_output.to_dict()['usage']["total_tokens"])
                }
                with open(RESULTS_PATH, "w") as f:
                    json.dump(gpt_response, f, indent=4)
                with open(TOKEN_COUNT_FILE, "w") as f:
                    json.dump(gpt_token_count, f, indent=4)