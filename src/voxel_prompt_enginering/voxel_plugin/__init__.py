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

"""
__init__.py.

This module defines a FiftyOne plugin operator for rerunning a language model (LLM)
inference on selected dataset samples with a custom prompt. It provides functionality
to process images, send them to a FastAPI-based LLM service, and store the generated
responses in the dataset.

Classes:
    RerunLLMOperator: A FiftyOne operator for reprocessing selected images with a
                      custom prompt using an LLM.

Functions:
    run_llm_on_image(image_path, prompt): Sends an image and prompt to the FastAPI
                                          vLLM service and retrieves the response.
    register(p): Registers the FiftyOne plugin operator.
"""

import base64

import fiftyone.operators as foo
import requests
from fiftyone.operators import types


class RerunLLMOperator(foo.Operator):
    """
    A FiftyOne operator for rerunning LLM inference on selected dataset samples with a custom prompt.

    Methods:
        config: Returns the configuration for the operator, including its name,
                label, and description.
        resolve_input: Defines the input form for the operator, allowing users
                       to specify a custom prompt.
        execute: Executes the LLM inference on selected samples, updates the
                 dataset with the prompt and response, and refreshes the UI.
        resolve_output: Defines the output response structure for the operator.
    """

    @property
    def config(self):
        """Returns the configuration for the operator."""
        return foo.OperatorConfig(
            name="rerun_llm",
            label="Rerun LLM with Custom Prompt",
            description="Enter a prompt and reprocess selected images with LLM.",
            dynamic=True,  # Enables live updates
            disable_schema_validation=True,
        )

    def resolve_input(self, ctx):
        """Define the user input form for the operator.

        Args:
            ctx: The FiftyOne context object.
        Returns:
            types.Property: A property object defining the input form structure.
        """
        inputs = types.Object()
        inputs.str("prompt", label="Enter prompt", required=True, default="")
        return types.Property(inputs, view=types.View(label="Enter LLM Prompt"))

    def execute(self, ctx):
        """Execute the LLM inference on selected dataset samples and updates the dataset.

        Args:
            ctx: The FiftyOne context object containing the selected samples and user input.
        Raises:
            ValueError: If no samples are selected.
        Returns:
            dict: A dictionary containing a status message indicating the completion of the process.
        """
        if not ctx.selected:
            raise ValueError("No samples selected!")

        prompt = ctx.params["prompt"]
        # Get or initialize the global prompt version
        prompt_version = ctx.dataset.info.get("prompt_version", 0) + 1
        ctx.dataset.info["prompt_version"] = prompt_version  # Update the version

        # Loop over selected samples
        for sample_id in ctx.selected:
            sample = ctx.dataset[sample_id]
            image_path = sample.filepath
            # Run LLM inference
            generated_text = run_llm_on_image(image_path, prompt)
            # Store the global prompt version for all samples
            sample.set_field(f"prompt_{prompt_version}", prompt, create=True)
            sample.set_field(f"response_{prompt_version}", generated_text, create=True)
            # Save the sample to persist changes in the database
            sample.save()
        ctx.dataset.save()
        ctx.ops.reload_dataset()
        return {"message": "LLM inference completed for selected samples!"}

    def resolve_output(self, ctx):
        """
        Define the output response structure for the operator.

        Args:
            ctx: The FiftyOne context object.

        Returns:
            types.Property: A property object defining the output response structure.
        """
        outputs = types.Object()
        outputs.str("message", label="Status")
        return types.Property(outputs, view=types.View(label="LLM Processing"))


def run_llm_on_image(image_path, prompt):
    """
    Send an image and a text prompt to the FastAPI vLLM service for inference.

    Args:
        image_path (str): The file path to the image to be processed.
        prompt (str): The text prompt to guide the language model's response.

    Returns:
        str: The generated text response from the vLLM service.
    """
    try:

        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {"image": image_base64, "prompt": prompt}
        response = requests.post("http://localhost:5001/infer", json=payload, timeout=30)
        return response.json().get("response", "")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return f"Request failed: {e}"

    except ValueError as e:
        print(f"[ERROR] Failed to parse JSON response: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response received'}")
        return f"Failed to parse JSON: {e}"

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return f"Unexpected error: {e}"


def register(p):
    """Register the FiftyOne plugin."""
    p.register(RerunLLMOperator)
