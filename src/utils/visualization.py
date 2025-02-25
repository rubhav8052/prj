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

import cv2
import numpy as np

# Step 1: Read the original image
image = cv2.imread("/home/ehc3kor/llm-rag-pipeline/data/gt_images/bulb_28.jpg")

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Canny edge detection
otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
gray = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)
# edges = cv2.Canny(gray, int(otsu_threshold), int(otsu_threshold*1.5))

# Step 4: Create a color mask from edges
# Create a 3-channel mask
mask = np.zeros_like(image)
mask[edges != 0] = [255, 255, 255]  # Set edges to white

# Step 5: Overlay the original image and the mask
overlay = cv2.addWeighted(image, 1, mask, 1, 0)

while True:
    cv2.imshow("Original Image", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("Overlay", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()