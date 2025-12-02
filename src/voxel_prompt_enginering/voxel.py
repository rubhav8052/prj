import os
import argparse
import fiftyone as fo
import fiftyone.plugins as fop
import time
#fop.reload_plugins()
#print(fop.list_plugins())

def main():
    parser = argparse.ArgumentParser(description="Load images into a FiftyOne dataset")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing images"
    )
    args = parser.parse_args()

    images_path = args.images

    # Print available plugins
    for plugin in fop.list_plugins():
        print(plugin.name, plugin.operators)
    print("Plugins directory:", fo.config.plugins_dir)

    # Dataset name
    dataset_name = "Prompt Playground"

    # Delete if exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    # Create dataset
    dataset = fo.Dataset(dataset_name)

    # Add images
    for img_file in os.listdir(images_path):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(images_path, img_file)
            sample = fo.Sample(filepath=filepath)
            dataset.add_sample(sample)

    # Launch FiftyOne App
    session = fo.launch_app(dataset, address="localhost", port=5151)
    #if running on compute instance:
    #session = fo.launch_app(dataset, address="0.0.0.0", port=5151)
    session.wait()

    time.sleep(100000)

if __name__ == "__main__":
    main()