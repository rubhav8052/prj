import logging
import os
import time
from pathlib import Path
from typing import List, Union
import pandas as pd

from azure.storage.blob import BlobServiceClient
import sys
sys.path.append("src")

from authenticate.auth import Authenticator
from mdm.mdd import MetaDataDumpClient
from mdm.tds import TrustedDataStorageClient
from mdm_viper_lib.mdm_sdk_extension.v2.entry import CachedEntryClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
os.environ["MDM_MDD_URI"] = "https://data-delivery-api.ad-alliance.biz/mddump"
os.environ["MDM_TDS_URI"] = "https://data-delivery-api.ad-alliance.biz/mdtds"
os.environ["MDM_AUTH_URI"] = "https://data-delivery-api.ad-alliance.biz/mdmauth"
os.environ["MDM_TDS_AUTH_SCOPE"] = "api://sp-pace-mdtds-pace-westeurope/.default"
os.environ["MDM_MDD_AUTH_SCOPE"] = "api://sp-pace-mddump-pace/.default"

credential = Authenticator().run()

tds_url = os.environ["MDM_TDS_URI"]
tds_client = TrustedDataStorageClient(tds_url,org_id="pace",credential=credential, credential_scopes=[os.environ["MDM_TDS_AUTH_SCOPE"]])

mdd_url = os.environ.get("MDM_MDD_URI")

mdd_client = MetaDataDumpClient(credential=credential, base_url=mdd_url,credential_scopes=[os.environ["MDM_MDD_AUTH_SCOPE"]],
    gateway=True,
    headers={"apikey": os.environ["SnF_API_Key"]},
    org_id="pace"
)

STORAGE_ACCOUNT_NAME = "vdeepingestprod"
CONTAINER_NAME = "datasets"
FOLDER_IN_BLOB = "attribute_labeling/tl_inlays2/"
account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
cred = DefaultAzureCredential(exclude_environment_credential=True)
blob_service_client = BlobServiceClient(account_url=account_url, credential=cred)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


class MDMFrameDownloader:
    def __init__(self, container_client, max_workers: int = 10):
        self.container_client = container_client
        self.download_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.upload_executor = ThreadPoolExecutor(max_workers=10)

    def upload_image_to_blob(self, image_data: bytes, folder: str, filename: str):
        try:
            blob_path = f"{FOLDER_IN_BLOB}{folder.strip('/')}/{filename}"
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(image_data, overwrite=True, blob_type="BlockBlob", max_concurrency=4)
            print(f"Uploaded to blob: {blob_path}")
        except Exception as e:
            logging.error(f"Upload failed for {filename}: {e}")

    def download_image(self, sha: str):
        try:
            entry_client = CachedEntryClient(
                sha,
                mdd_client=mdd_client,
                tds_client=tds_client,
                credential=credential
            )
            entry_client.file.load_in_ram()
            return sha, entry_client.file.file_content
        except Exception as e:
            logging.error(f"Failed to download SHA {sha}: {e}")
            return sha, None

    def download_files_parallel(self, shas: List[str]) -> List[tuple]:
        futures = [self.download_executor.submit(self.download_image, sha) for sha in shas]
        results = []
        for future in as_completed(futures):
            sha, image_data = future.result()
            if image_data:
                results.append((sha, image_data))
        return results

    def upload_files_parallel(self, images: List[tuple], folder: str):
        futures = [
            self.upload_executor.submit(self.upload_image_to_blob, image_data, folder, f"{sha}.png")
            for sha, image_data in images
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Upload error: {e}")

    def __call__(self, frame_sha_input: Union[List[str], Path], output_dir: str, batch_size: int = 100):
        if isinstance(frame_sha_input, Path):
            with open(frame_sha_input, "r") as fh:
                frame_shas = [line.strip() for line in fh if line.strip()]
        else:
            frame_shas = frame_sha_input

        n_shas = len(frame_shas)
        now = time.time()
        try:
            for i in range(0, n_shas, batch_size):
                batch_shas = frame_shas[i:i+batch_size]
                images = self.download_files_parallel(batch_shas)
                self.upload_files_parallel(images, output_dir)
                print(f"Completed batch {i//batch_size + 1}: {len(images)} uploaded")
            print(f"Downloaded and uploaded {n_shas} blobs in parallel: {time.time() - now:.2f}s")
        except Exception as e:
            logging.error(f"Download/upload error: {e}")
            raise e

if __name__ == "__main__":

    downloader = MDMFrameDownloader(container_client)
    df = pd.read_csv("full_dataset.csv")

    for sub_class in df["inlay"].unique():
        shas = df[df["inlay"] == sub_class]["frame_sha"].tolist()
        downloader(shas, sub_class)
