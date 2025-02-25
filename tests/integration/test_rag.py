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


def rag(search_results_path: str) -> list[str]:
    rag = []
    d = {}
    color_list, inlay_list = [], []
    with open(search_results_path, "r") as f:
        data = json.load(f)
    for file, val in data.items():
        if file == "TrafficLightBulb_1272.jpg":
            for key, value in val["topk_matches"].items():
                col, inl = value["color"], value["inlay"]
                color_list.append(col), inlay_list.append(inl)
            d[file] = {"color": color_list, "inlay": inlay_list}
            rag.append(d)

    for ele in rag:
        for _, val in ele.items():
            print(val["color"])

    # return rag


print(rag("/home/mgx2kor/Desktop/langchain/llm-rag-pipeline/simsearch/search_results_canny_otsu_mask.json"))