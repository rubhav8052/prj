# ============================================================
#  C O P Y R I G H T
# ------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
# 
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ============================================================

import os
import shutil
import argparse

def copy_dir_skip_existing(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        # Recreate directory structure
        rel_path = os.path.relpath(root, src)
        dest_path = os.path.join(dst, rel_path)
        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_path, file)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

# Add argument parsing
parser = argparse.ArgumentParser(description="Launch FiftyOne session with specified source directory and timeout.")
parser.add_argument('--source-dir', type=str, required=True, help='Name of the source directory in os.environ["AZUREML_DATAREFERENCE_workspaceblobstore"]+/attribute_labeling/fiftyone/db/')
parser.add_argument('--timeout', type=int, default=1800, help='Session timeout in seconds (default: 1800)') # Default to 30*60 = 1800

args = parser.parse_args()
source_dir_name = args.source_dir
session_timeout = args.timeout

os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = '/fiftyone/dataset/'
os.environ["FIFTYONE_DATABASE_DIR"] = '/fiftyone/db/'
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = '/fiftyone/dataset/'

os.symlink(os.environ['AZUREML_DATAREFERENCE_workspaceblobstore']+'/attribute_labeling', '/attribute_labeling')

copy_dir_skip_existing('/attribute_labeling/fiftyone/dataset/', '/fiftyone/dataset/')
copy_dir_skip_existing(f'/attribute_labeling/fiftyone/db/{source_dir_name}', '/fiftyone/db/')
copy_dir_skip_existing('/attribute_labeling/fiftyone/dataset/', '/fiftyone/dataset/')

import fiftyone as fo
session = fo.launch_app(port=3000, address='0.0.0.0')

import subprocess
subprocess.Popen("pip install azureml-core".split(' ')).wait()
from azureml.core import Run

run = Run.get_context()
details = run.get_details()
or_url = details["services"]["Jupyter"]["endpoint"].replace('jpytr', '3000')

run.description = or_url
session.wait(session_timeout)
