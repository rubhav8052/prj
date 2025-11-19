__copyright__ = """
===================================================================================
 C O P Y R I G H T
-----------------------------------------------------------------------------------
 Copyright (c) 2023-2025 Robert Bosch GmbH and Cariad SE. All rights reserved.
===================================================================================
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import fiftyone as fo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleanup old FiftyOne datasets for voxel teams"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        action="append",
        default=["tl_inlay", "dyo_vehicle_type"],
        help=(
            "Dataset name prefix to consider for deletion. "
            "Can be passed multiple times. "
            "Default: tl_inlay, dyo_vehicle_type"
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete datasets older than this many days (default: 30)",
    )
    return parser.parse_args()


def ensure_fiftyone_env() -> None:
    uri = os.getenv("FIFTYONE_API_URI")
    key = os.getenv("FIFTYONE_API_KEY")
    if not uri or not key:
        raise RuntimeError(
            "FIFTYONE_API_URI and FIFTYONE_API_KEY must be set in the environment."
        )
    logger.info("Using FiftyOne Teams at %s", uri)


def cleanup_datasets(prefixes: list[str], max_age_days: int) -> list[str]:
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    deleted: list[str] = []

    logger.info(
        "Starting cleanup: prefixes=%s, max_age_days=%d, cutoff=%s",
        prefixes,
        max_age_days,
        cutoff.isoformat(),
    )

    for name in fo.list_datasets():
        if not any(name.startswith(p) for p in prefixes):
            continue

        ds = fo.load_dataset(name)

        # Only touch persistent datasets
        if not getattr(ds, "persistent", False):
            logger.info("Skipping non-persistent dataset %s", name)
            continue

        created = getattr(ds, "created_at", None)
        if created is None:
            logger.info("Dataset %s has no created_at; skipping", name)
            continue

        # Normalize timezone for comparison
        if getattr(created, "tzinfo", None) is not None:
            created_naive = created.replace(tzinfo=None)
        else:
            created_naive = created

        if created_naive < cutoff:
            logger.info(
                "Deleting dataset '%s' (created_at=%s)", name, created.isoformat()
            )
            fo.delete_dataset(name)
            deleted.append(name)
        else:
            logger.info(
                "Keeping dataset '%s' (created_at=%s, newer than cutoff)",
                name,
                created.isoformat(),
            )

    return deleted


def main() -> None:
    args = parse_args()
    ensure_fiftyone_env()

    deleted = cleanup_datasets(prefixes=args.prefix, max_age_days=args.max_age_days)

    if deleted:
        print("Deleted datasets:")
        for n in deleted:
            print(n)
    else:
        print("No datasets matched criteria; nothing deleted.")


if __name__ == "__main__":
    main()
