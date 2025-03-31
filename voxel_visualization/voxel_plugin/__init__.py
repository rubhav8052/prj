import fiftyone.operators as foo
import fiftyone.operators.types as types
import requests
import base64


class RerunLLMOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="rerun_llm",
            label="Rerun LLM with Custom Prompt",
            description="Enter a prompt and reprocess selected images with LLM.",
            dynamic=True,  # Enables live updates
            disable_schema_validation= True,
        )
  
    def resolve_input(self, ctx):
        """Defines the user input form"""
        inputs = types.Object()
        inputs.str("prompt", label="Enter prompt", required=True, default="")
        return types.Property(inputs, view=types.View(label="Enter LLM Prompt"))

    
    def execute(self, ctx):
        """Executes the LLM inference and updates dataset."""
        if not ctx.selected:
            raise ValueError("No samples selected!")

        prompt = ctx.params["prompt"]
        # Loop over selected samples
        for sample_id in ctx.selected:
            sample = ctx.dataset[sample_id]  # Retrieve sample from dataset
            image_path = sample.filepath  # Get the image path

            # Run LLM inference
            generated_text = run_llm_on_image(image_path, prompt)
            # Find the next available index for storing prompt-response history
            index = 1
            # Ensure we only increment if the field is both present and has a value
            while sample.has_field(f"prompt_{index}") and sample[f"prompt_{index}"] is not None:
                index += 1

            # Store the prompt and response dynamically
            sample.set_field(f"prompt_{index}", prompt, create=True)
            sample.set_field(f"response_{index}", generated_text, create=True)

            # Save the sample to persist changes in the database
            sample.save()

        ctx.ops.reload_dataset()  # Refresh dataset UI
        return {"message": "LLM inference completed for selected samples!"}

    def resolve_output(self, ctx):
        """Defines the output response"""
        outputs = types.Object()
        outputs.str("message", label="Status")
        return types.Property(outputs, view=types.View(label="LLM Processing"))
  
def run_llm_on_image(image_path, prompt):
    """Calls the FastAPI vLLM service"""
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {"image": image_base64, "prompt": prompt}
    response = requests.post("http://localhost:5001/infer", json=payload)
    return response.json().get("response", "")

def register(p):
    """Registers the FiftyOne plugin"""
    p.register(RerunLLMOperator)