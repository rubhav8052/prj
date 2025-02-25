# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2025 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================

import json

from config.rag_config import TOPK


class Metrics:
    def __init__(self):
        pass

    def calc_metrics(self, path: str) -> list[int]:
        """Calculates the accuracy values for the desired attributes
        Args:
            path (str): Path to the results.json file.
        Returns:
            List[int]: The list with accuracy metrics for the desied attributes.
        """
        with open(path, "r") as f:
            data = json.load(f)

        total_count = len(data)
        global_color_count, global_inlay_count = 0, 0
        color_consistency, inlay_consistency = 0, 0

        for _, val in data.items():
            local_color_count, local_inlay_count = 0, 0
            gt_inlay = val["test_attributes"][0]["inlay"]
            gt_color = val["test_attributes"][0]["color"]

            for _, value in val["topk_matches"].items():
                local_color_count += gt_color == value["color"]
                local_inlay_count += gt_inlay == value["inlay"]

            global_color_count += local_color_count > 0
            global_inlay_count += local_inlay_count > 0
            color_consistency += local_color_count / TOPK
            inlay_consistency += local_inlay_count / TOPK

        color_accuracy = global_color_count / total_count
        inlay_accuracy = global_inlay_count / total_count
        color_consistency = color_consistency / total_count
        inlay_consistency = inlay_consistency / total_count

        return {
            "Color Accuracy": color_accuracy,
            "Color Consistency": color_consistency,
            "Inlay Accuracy": inlay_accuracy,
            "Inlay Consistency": inlay_consistency,
        }