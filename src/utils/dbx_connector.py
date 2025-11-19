"""Connector for Azure Databricks."""

__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023-2025 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""
import os
from omegaconf import DictConfig
from types import TracebackType
from typing import Optional

from azure.identity import DefaultAzureCredential
from databricks import sql
import datetime


class DBXConnector:
    """Connector for Azure Databricks."""

    # This is universal for all Databricks instances and Warehouses
    DATABRICKS_AZURE_RESOURCE_ID = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"

    def __init__(
        self,
        credential: DefaultAzureCredential,
        databricks_workspace_hostname: str = os.environ.get(
            "DATABRICKS_BASE_URL", "adb-8617216030703889.9.azuredatabricks.net"
        ),
        sql_warehouse_path: str = os.environ.get("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/72567d2a4be41c46"),
    ) -> None:
        """Initialize the Databricks connector.

        Args:
            credential: The Azure credential.
            databricks_workspace_hostname: The hostname of the Databricks workspace.
            sql_warehouse_path: The path to the SQL warehouse.
        """
        self.credential = credential
        self.databricks_workspace_hostname = databricks_workspace_hostname
        self.sql_warehouse_path = sql_warehouse_path

    def __enter__(self) -> "DBXConnector":
        """Enter the context manager and establish the Databricks connection."""
        self.token = self.credential.get_token(self.DATABRICKS_AZURE_RESOURCE_ID)
        self.connection = sql.connect(
            server_hostname=self.databricks_workspace_hostname,
            http_path=self.sql_warehouse_path,
            access_token=self.token.token,
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit the context manager and close the Databricks connection.

        Args:
            exc_type: The type of the exception.
            exc_value: The exception value.
            traceback: The traceback.

        Returns:
            None
        """
        self.connection.close()

    def fetch_from_dbx_by_date(
            self,
            cfg: DictConfig,
            url_columns: list[str],
            bbox_columns: list[str],
            date: datetime.datetime,
            timeout_seconds: int = 30,
    ):
        url_cols_sql = ",\n        ".join(f"f.{c}" for c in url_columns)
        bbox_cols_sql = ",\n        ".join(f"o.{c}" for c in bbox_columns)
        with self.connection.cursor() as cursor:
            query = f"""
                select
                    o.{cfg.label_column},
                    {url_cols_sql},
                    {bbox_cols_sql},
                    case
                        when cardinality(s.involved_models) > 0 then
                            element_at(
                            split(
                                element_at(s.involved_models, -1),
                                '/'
                            ),
                            -1
                            )
                        else null
                    end as last_involved_model                    
                from 
                    bronze.per_gtop_automated_labels.{cfg.mutti_object_table} o
                join 
                    bronze.per_gtop_automated_labels.{cfg.mutti_frame_table} f
                    on 
                        o.label_request_id = f.label_request_id
                    and
                        o.frame_master_index = f.frame_master_index
                join
                    bronze.per_gtop_automated_labels.{cfg.mutti_sequence_table} s
                    on
                        f.sequence_label_request_id = f.sequence_label_request_id                        
                where 
                    cast(o.processed_at as date) = '{date.isoformat()}'
                limit {getattr(cfg, "limit_rows", 100)};
                """
            cursor.execute(f"SET STATEMENT_TIMEOUT={timeout_seconds}")
            cursor.execute(query)
            df = cursor.fetchall_arrow().to_pandas()
        return df
    
    def fetch_from_dbx_by_lrid(
            self,
            cfg: DictConfig,
            url_columns: list[str],
            bbox_columns: list[str],
            lr_ids: list[str],
            timeout_seconds: int = 30,
    ):
        url_cols_sql = ",\n        ".join(f"f.{c}" for c in url_columns)
        bbox_cols_sql = ",\n        ".join(f"o.{c}" for c in bbox_columns)
        lr_id_placeholders = ",".join("?" for _ in lr_ids)
        with self.connection.cursor() as cursor:
            query = f"""
                select
                    o.{cfg.label_column}, 
                    {url_cols_sql},
                    {bbox_cols_sql},
                    case
                        when cardinality(s.involved_models) > 0 then
                            element_at(
                            split(
                                element_at(s.involved_models, -1),
                                '/'
                            ),
                            -1
                            )
                        else null
                    end as last_involved_model
                from 
                    bronze.per_gtop_automated_labels.{cfg.mutti_object_table} o
                join 
                    bronze.per_gtop_automated_labels.{cfg.mutti_frame_table} f
                    on 
                        o.label_request_id = f.label_request_id
                    and
                        o.frame_master_index = f.frame_master_index
                join
                    bronze.per_gtop_automated_labels.{cfg.mutti_sequence_table} s
                    on
                        f.sequence_label_request_id = f.sequence_label_request_id
                where 
                    o.label_request_id in ({lr_id_placeholders})
                limit {cfg.limit_rows};
                """
            cursor.execute(f"SET STATEMENT_TIMEOUT={timeout_seconds}")
            cursor.execute(query, lr_ids)
            df = cursor.fetchall_arrow().to_pandas()
        return df    
