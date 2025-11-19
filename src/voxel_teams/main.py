__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023-2025 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import os, sys
import datetime
import hydra
import logging
from omegaconf import DictConfig
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential
sys.path.append("src")
from utils.dbx_connector import DBXConnector
from create_voxel_dataset import make_usecase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_azure_credentials() -> ChainedTokenCredential:
    """Return Azure credentials.

    The chain first checks for the AzureCliCredential (mostly used in local development),
    and then falls back to the DefaultAzureCredential. DefaultAzureCredential will try
    all authentication methods in order, particularly EnvironmentCredential (used in Argo) and
    ManagedIdentityCredential (used in CI/CD).
    """
    credentials = ChainedTokenCredential(
        AzureCliCredential(), DefaultAzureCredential(managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"))
    )
    return credentials


def fetch_mutti_objects(cfg: DictConfig, url_columns: list[str], bbox_columns: list[str], date: datetime.date, label_request_ids: list[str]):
    """
    Fetch objects from a Mutti object table, filtered either by date or by label_request_id.
    Only one of the filters should be provided.
    """
    with DBXConnector(credential=get_azure_credentials()) as connector:
        if date:
            objects = connector.fetch_from_dbx_by_date(
                cfg,
                url_columns,
                bbox_columns,
                date, 
                300
            )
        else:
            objects = connector.fetch_from_dbx_by_lrid(
                cfg,
                url_columns,
                bbox_columns,
                label_request_ids, 
                300
            )
    return objects

def standardize_df(cfg: DictConfig):
    url_columns = []
    bbox_columns = []
    for stream in cfg.streams:
        url_columns.append(f"{stream.lower()}_rectified_frame_url")
        bbox_columns.append(f"{stream}_bbox")
    return url_columns, bbox_columns


def _parse_lr_ids_from_cfg(cfg: DictConfig) -> list[str]:
    """
    Expects cfg.label_request_ids to be a string.
    """
    label_request_ids = cfg.get("label_request_ids", "") or ""
    return [s.strip() for s in label_request_ids.split(",") if s.strip()]

@hydra.main(version_base="1.3", config_path="configs", config_name="tl_inlay")
def main(cfg: DictConfig) -> None:
    """
    Behavior:
    - If cfg.label_request_ids is empty (""):
        -> default mode: date = yesterday, label_request_ids = []
    - If cfg.label_request_ids is non-empty:
        -> date = None, label_request_ids = parsed list from cfg.lr_ids
    """
    label_request_ids = _parse_lr_ids_from_cfg(cfg)
    if label_request_ids:
        # LR-only mode
        date = None
    else:
        # Default mode: run for yesterday
        date = datetime.date.today() - datetime.timedelta(days=1)
        label_request_ids = []

    if date is None and not label_request_ids:
        raise ValueError("Provide either a date (default) or at least one label_request_id.")

    if len(label_request_ids) > 10:
        raise ValueError("Cannot process more than 10 label_request_ids.")

    url_columns, bbox_columns = standardize_df(cfg)

    objects_df = fetch_mutti_objects(
        cfg,
        url_columns=url_columns,
        bbox_columns=bbox_columns,
        date=date,
        label_request_ids=label_request_ids,
    )
    logger.info(f"Fetched {len(objects_df)} objects from Mutti.")

    if not objects_df.empty:
        uc = make_usecase(cfg, objects_df, date, zip(url_columns, bbox_columns))
        uc(uc.df)


if __name__ == "__main__":
    main()
