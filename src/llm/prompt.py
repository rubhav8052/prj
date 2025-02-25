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

        prompt = f"""The image is the bulb of a traffic light. Fetch me information about the color and inlay of traffic bulb in the image. Color is defined as the color of the corresponding traffic light bulb or lane control light. If it is confusing between Red and Yellow, choose Yellow only when it has distinct Yellow tint, otherwise choose Red. same logic for Green and Yellow if confused. Possible values for color are {color_list}. Inlay is defined as the shape of the corresponding traffic light bulb. If there is a single arrow in the image, then choose between ArrowStraight (if arrow is facing upwards), ArrowLeft (if arrow is facing left), ArrowRight (if arrow is facing right). If an arrow is tilted and not vertical or horizontal, then it might be one among ArrowRight45Up, if arrow is pointing right or ArrowLeft45Up, if arrow is pointing left. If there are two arrows in the bulb, it could either ArrowStraightLeft, if left is present or ArrowStraightRight, if right is present. Possible values for inlay are {inlay_list}. Reverify output by checking the image again. Generate only JSON in the following format: {{\"TrafficLightBulb\": {{\"color\": \"{{value1}}\", \"inlay\": \"{{value2}}\"}}}} Here {{value1}} will be replaced color value and {{value2}} will be replaced by inlay value."""