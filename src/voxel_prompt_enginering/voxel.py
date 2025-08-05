import os
import fiftyone as fo
import fiftyone.plugins as fop
import time
#fop.reload_plugins()
#print(fop.list_plugins())
for plugin in fop.list_plugins():
    print(plugin.name, plugin.operators)
print(fo.config.plugins_dir)

images_path =  "./Right_Of_Way_frames" # Change Path to folder containing frames

# Create a new dataset (or load if it already exists)
dataset_name = "Right Of Way POC"
if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
dataset = fo.Dataset(dataset_name)

# Add all images from the directory
for img_file in os.listdir(images_path):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(images_path, img_file)
        sample = fo.Sample(filepath=filepath)
        dataset.add_sample(sample)

# Launch FiftyOne App

session = fo.launch_app(dataset, address="localhost", port=5151)
session.wait()

time.sleep(100000)
