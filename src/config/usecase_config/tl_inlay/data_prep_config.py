import os

from src.config.data_prep_config import DataPrepConfig
    
config = DataPrepConfig()
config.input_csv_paths = [os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/attribute_labeling/tl_inlays/full_dataset.csv']
# Base path for resolving relative image paths within CSVs
config.image_base_dir = os.environ['AZUREML_DATAREFERENCE_VDEEPINGESTPROD']+'/attribute_labeling/tl_inlays/'
# Directory to save output JSONL files
config.output_dir = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling/tl_inlays/datasets/drawn_green_box_with_context/'
# Column names from CSV
config.attribute_col = "inlay"
# Prompt file for attribute task (used if generating attribute dataset)
config.attribute_prompt_file = os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+f'/attribute_labeling/tl_inlays/datasets/drawn_green_box_with_context/prompts/traffic_lights_inlay_prompt.md' 
config.img_sha_col = "frame_sha"
config.test_size = 0.10
#ignore crops having height/width lower than this
min_height_width_pixel = 32

# Output filenames
config.attribute_train_file = "tl_inlay_train_v1.jsonl"
config.attribute_val_file = "tl_inlay_val_v1.jsonl"
config.attribute_test_file = "tl_inlay_test_v1.jsonl"