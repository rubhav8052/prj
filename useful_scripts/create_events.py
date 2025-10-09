import hashlib
import json
import uuid
import os
import sys
import csv
from datetime import datetime
import random
import string

import bosch_openlabel
from mdm_viper_lib.dataloop_event_lib.enums import Environment
from mdm_viper_lib.dataloop_event_lib.message_router import MessageRouter
from mdm.tds import TrustedDataStorageClient
from mdm_viper_lib.mdm_sdk_extension.v2.entry import CachedEntryClient
from dotenv import load_dotenv
from mdm.mdd import MetaDataDumpClient
from bosch_openlabel.model.v2 import BoschOpenLabelFormat as BoschOpenLabelFormatV2
from mdm_viper_lib.dataloop_integration.helper import set_mdm_environment
from pace.events.publishers.autolabeling.tracking_completed import TrackingCompletedEvent, TrackingCompleted

# Add src to sys.path if not already there
if "src" not in sys.path:
    sys.path.append("src")
from authenticate.auth import Authenticator

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants (read directly from environment variables) ---
# Use os.getenv with a default fallback for robustness
MDM_MDD_URI = os.getenv("MDM_MDD_URI", "https://data-delivery-api.ad-alliance.biz/mddump")
MDM_TDS_URI = os.getenv("MDM_TDS_URI", "https://data-delivery-api.ad-alliance.biz/mdtds")
MDM_AUTH_URI = os.getenv("MDM_AUTH_URI", "https://data-delivery-api.ad-alliance.biz/mdmauth")
MDM_TDS_AUTH_SCOPE = os.getenv("MDM_TDS_AUTH_SCOPE", "api://sp-pace-mdtds-pace-westeurope/.default")
MDM_MDD_AUTH_SCOPE = os.getenv("MDM_MDD_AUTH_SCOPE", "api://sp-pace-mddump-pace/.default")
SNF_API_KEY = os.getenv("SnF_API_Key")
ORG_ID = "pace"
CURRENT_SERVICE = "sequencing"
# Determine the queue environment once, based on the MDM_ENVIRONMENT variable
# This ensures consistency for Dataloop events as well
QUEUE_ENVIRONMENT_STR = os.getenv("MDM_ENVIRONMENT", "prod")
QUEUE_ENVIRONMENT = Environment[QUEUE_ENVIRONMENT_STR] # Converts string to Enum member

# Set the global MDM environment for SDK functions that rely on it
set_mdm_environment(QUEUE_ENVIRONMENT_STR)

# --- Client Initialization ---
try:
    credential = Authenticator().run()
except Exception as e:
    print(f"Error during authentication: {e}")
    sys.exit(1)

tds_client = TrustedDataStorageClient(
    base_url=MDM_TDS_URI,
    org_id=ORG_ID,
    credential=credential,
    credential_scopes=[MDM_TDS_AUTH_SCOPE]
)


mdd_client = MetaDataDumpClient(
    credential=credential,
    base_url=MDM_MDD_URI,
    credential_scopes=[MDM_MDD_AUTH_SCOPE],
    gateway=True,
    headers={"apikey": SNF_API_KEY}, # Ensure SNF_API_KEY is not None
    org_id=ORG_ID
)

message_router = MessageRouter.create(
    current_service=CURRENT_SERVICE,
    env=QUEUE_ENVIRONMENT,
    credential=credential
)

def create_alf(olf_sha: str) -> BoschOpenLabelFormatV2:
    """
    Retrieves and decodes an ALF (Asset Labeling Format) from MDM.

    Args:
        olf_sha: The SHA256 hash of the OpenLabelFormat file.

    Returns:
        The decoded BoschOpenLabelFormat content as a dictionary.

    Raises:
        FileNotFoundError: If the ALF cannot be loaded.
        json.JSONDecodeError: If the content is not valid JSON.
    """
    entry = CachedEntryClient(
        olf_sha,
        mdd_client=mdd_client,
        tds_client=tds_client,
        credential=credential
    )  
    entry.file.load_in_ram()
    bolf_content = entry.file.file_content   
    bolf = bosch_openlabel.read(json.loads(bolf_content.decode("utf-8")), run_upgrade=True)
    return bolf

def create_message(alf: BoschOpenLabelFormatV2, split_sha:str) -> TrackingCompletedEvent:
    if not hasattr(alf.openlabel.metadata, "label_request_id") or \
        alf.openlabel.metadata.label_request_id is None or \
        split_sha != alf.openlabel.metadata.label_request_id[:64]:

        salt = "".join(random.choices(string.ascii_letters + string.digits, k=5)).lower()
        alf.openlabel.metadata.label_request_id = f"{split_sha}fc1f00000tlautolabel{salt}"

    label_request_id = alf.openlabel.metadata.label_request_id

    payload = TrackingCompleted(
        label_usecase="tl_autolabel",
        label_request_id=label_request_id,
        alf=alf.dict(),
        priority=2
    )

    tracking_completed = TrackingCompletedEvent(
        specversion="1.0",
        source="test/local",  # Consider making this configurable or dynamic
        id=hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        time=datetime.now(),
        subject=f"/MDM/{ORG_ID}/labeling_usecase/{payload.label_usecase}",
        data=payload,
    )
    return tracking_completed

def process_csv_entry(entry_data: dict) -> None:
    """
    Processes a single entry (row) from the CSV file.
    """
    file_hash = entry_data.get('file_hash')
    split_sha = entry_data.get('split_hash')

    if not file_hash:
        print("Skipping row due to missing 'file_hash'")
        return
    alf_content = create_alf(file_hash)
    message = create_message(alf_content, split_sha)
    message_router(message, _skip_type_validation=True)


def main(csv_file_path: str) -> None:
    """
    Main function to read the CSV, iterate through entries, and process each.

    Args:
        csv_file_path: The path to the input CSV file.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        sys.exit(1)

    processed_count = 0
    failed_count = 0

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile) # DictReader maps rows to dictionaries by header
        # Check if required headers are present
        required_headers = ['file_hash', 'split_hash']

        if not all(header in reader.fieldnames for header in required_headers):
            print(f"Error: CSV is missing one or more required headers. Found: {reader.fieldnames}, Required: {required_headers}")
            sys.exit(1)

        for row in reader:
            try:
                process_csv_entry(row)
                processed_count += 1
            except Exception as e:
                print(e)
                failed_count += 1
    
    print("\n--- Processing Summary ---")
    print(f"Total entries attempted: {processed_count + failed_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {failed_count}")

if __name__ == "__main__":
    input_csv_file = "tl_others.csv" #provide the csv filename
    main(input_csv_file)