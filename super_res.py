import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("FSRCNN_x4.pb")
sr.setModel("fsrcnn", 4)

old="5138"
img = cv2.imread(f"images/test_images/{old}.png")
sr_img = sr.upsample(img)

print(img.shape, "→", sr_img.shape)

# Resize to what DeepSeek-VL2 expects
sr_img = cv2.resize(sr_img, (224, 224))

cv2.imwrite(f"images/test_images/{old}_new.png", sr_img)
