__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023-2025 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

from typing import Any
from collections.abc import Iterable
from datetime import datetime
import logging
import os
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.management as fom
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaseVoxelFromDF:
    """
    Build a FiftyOne dataset from a DataFrame.
    """

    def __init__(self, cfg: DictConfig):

        if not os.getenv("FIFTYONE_API_URI") or not os.getenv("FIFTYONE_API_KEY"):
            raise RuntimeError("Set FIFTYONE_API_URI and FIFTYONE_API_KEY in your environment")

        self.cfg = cfg
        self.usecase: str = cfg.usecase
        if not self.date:
            lr_ids = [s.strip() for s in cfg.label_request_ids.split(",") if s.strip()]
            self.lr_id_name = lr_ids[0][-5:]

    # ---------- dataset naming ----------
    @property
    def create_dataset_name(self) -> str:
        if self.date is not None:
            return f"{self.usecase}_{self.model}_{str(self.date)}"
        return f"{self.usecase}_{self.model}_{self.lr_id_name}"

    # ---------- hooks you can override ----------
    def extra_fields_for_detection(self, row_extras: dict[str, Any]) -> dict[str, Any]:
        """
        Return a dict of fields to add on each fo.Detection from this row's extras.
        Default: return all extras unchanged (attach all extras on detection).
        """
        return dict(row_extras)

    def apply_sample_extras(self, sample: fo.Sample, grouped_extras: list[dict[str, Any]]) -> None:
        """
        Attach any aggregated extras to the sample (one per image).
        Default: do nothing. Override in child classes.
        """
        return

    def __call__(self, df: pd.DataFrame) -> fo.Dataset:
        """
        Build a FiftyOne dataset from a DataFrame with URL + bbox columns.

        Assumes:
        - df has at least ['class_column', 'fc1_rectified_frame_url', 'FC1_bbox']
        - self.zipped_cols is a list of (url_col, bbox_col) pairs that
            specify which camera/url columns are present for this usecase.

        Behavior:
        - One sample per distinct URL (across all configured URL columns)
        - Detections are added/merged per URL
        - FC1 images use 6090 x 3074, TV images use 1536 x 1536
        - Camera & class are exposed at both sample and detection levels
        """
        name = self.create_dataset_name

        if name in fo.list_datasets():
            fo.delete_dataset(name)

        ds = fo.Dataset(name, persistent=True)

        for tag in (self.usecase, self.model):
            if tag not in ds.tags:
                ds.tags.append(tag)
        ds.save()

        def _camera_from_url_col(col: str) -> str:
            return col.split("_rectified")[0]

        # To avoid repeatedly matching on the same URL across different columns,
        # we maintain a small cache of URL -> sample mapping
        sample_cache = {}

        for url_col, bbox_col in self.zipped_cols:
            if url_col not in df.columns or bbox_col not in df.columns:
                continue

            camera_name = _camera_from_url_col(url_col)
            # Group by URL in this column
            for url_val, g in df.groupby(url_col, dropna=False):
                url_str = str(url_val).strip()
                if not url_str:
                    continue

                dets = []
                for _, row in g.iterrows():
                    label = str(row["class_column"])
                    bbox = row[bbox_col]
                    if bbox is None:
                        continue

                    x_center, y_center, w, h = map(float, bbox)

                    if x_center < 0 or y_center < 0:
                        continue

                    # Image size depends on camera
                    if camera_name.lower().startswith("fc1"):
                        image_width, image_height = 6090, 3074
                    else:
                        image_width, image_height = 1536, 1536

                    # Convert to normalized [xmin, ymin, width, height]
                    xmin = (x_center - w / 2.0) / image_width
                    ymin = (y_center - h / 2.0) / image_height
                    width = w / image_width
                    height = h / image_height

                    det = fo.Detection(
                        label=label,
                        bounding_box=[xmin, ymin, width, height],
                        camera=camera_name,
                        tags=[camera_name, label],
                    )
                    dets.append(det)

                if not dets:
                    continue

                # --- Create or update sample for this URL ---
                if url_str in sample_cache:
                    sample = sample_cache[url_str]
                else:
                    view = ds.match(F("filepath") == url_str)
                    if len(view) > 0:
                        sample = view.first()
                    else:
                        sample = fo.Sample(filepath=url_str)
                        ds.add_sample(sample, expand_schema=True)
                    sample_cache[url_str] = sample

                # Merge detections across cameras / columns
                existing_dets = []
                if "detections" in sample and sample["detections"] is not None:
                    existing_dets = list(sample["detections"].detections)

                all_dets = existing_dets + dets
                sample["detections"] = fo.Detections(detections=all_dets)

                # ---- Sample-level fields for image-level filtering ----
                cameras = sorted(
                    {
                        getattr(d, "camera", None)
                        for d in all_dets
                        if getattr(d, "camera", None)
                    }
                )
                classes = sorted({d.label for d in all_dets if d.label})

                sample["cameras"] = cameras
                sample["classes"] = classes

                # Also tag the sample with cameras + classes
                sample_tags = set(sample.tags or [])
                sample_tags.update(cameras)
                sample_tags.update(classes)
                sample.tags = sorted(sample_tags)

                sample.save()

        ds.save()
        logger.info(f"Dataset {name} saved!")
        fom.set_dataset_user_group_permission(
            dataset_name=name,
            user_group="All_Users_Group",
            permission=fom.VIEW,
        )
        logger.info("Access provided!")


# -----------------------------
# Example child usecases
# -----------------------------
class TrafficLightInlayUC(BaseVoxelFromDF):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame, date: datetime, zipped_cols: Iterable[tuple[str, str]]):
        self.date = date
        super().__init__(cfg)       
        self.model = df["last_involved_model"][0]
        self.df = self._map_cols_to_objectcfg(df)
        self.zipped_cols = zipped_cols

    def _map_cols_to_objectcfg(self, df):
        df=df.rename(
            columns={
                "FC1_predicted_inlay":"class_column"
            }
        )
        return df

    def extra_fields_for_detection(self, row_extras: dict[str, Any]) -> dict[str, Any]:
        # Example: keep only a whitelist and normalize some names
        out = {}
        if "lr_id" in row_extras:
            out["lr_id"] = str(row_extras["lr_id"])
        return out
    
class DYOVehicleUC(BaseVoxelFromDF):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame, date: datetime, zipped_cols: Iterable[tuple[str, str]]):
        self.date = date
        super().__init__(cfg)
        self.model = df["last_involved_model"][0]
        self.df = self._map_cols_to_objectcfg(df)
        self.zipped_cols = zipped_cols

    def _map_cols_to_objectcfg(self, df):
        df=df.rename(
            columns={
            "fused_vehicle_type":"class_column"
            }
        )
        return df

# -----------------------------
# Factory
# -----------------------------
_REGISTRY: dict[str, type[BaseVoxelFromDF]] = {
    "tl_inlay": TrafficLightInlayUC,
    "dyo_vehicle_type": DYOVehicleUC
}

def make_usecase(
        cfg: DictConfig, 
        df: pd.DataFrame, 
        date: datetime, 
        zipped_cols: Iterable[tuple[str, str]]
    ) -> BaseVoxelFromDF:
    try:
        return _REGISTRY[cfg.usecase](cfg, df, date, zipped_cols)
    except KeyError:
        raise ValueError(f"Unknown usecase '{cfg.usecase}'. Known: {list(_REGISTRY)}")
