from azureml.core import Workspace, Model, Run

def register_aml_model(model_path: str, tags: dict, model_name: str):
    """
    Registers a model in Azure ML Studio from a blobstore path.

    Args:
        blobstore_path (str): Path to the model file in current experiment.
        tags (dict): Dictionary of tags for metadata.
        model_name (str): Name to register the model with.

    Returns:
        Model: The registered Azure ML model object.
    """
    # Load Azure ML Workspace
    ws = Run.get_context().experiment.workspace

    # Register the model
    model = Model.register(workspace=ws,
                           model_path=model_path,
                           model_name=model_name,
                           tags=tags)
    
    print(f"Model '{model.name}' registered successfully. Version: {model.version}")
    return model
