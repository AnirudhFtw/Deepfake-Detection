import numpy as np
import cv2

# Load DCT array
dct = np.load(r"C:\Honours\project\processed_dct\fake\deepfakes_0\frame_0_0.npy")

# Take magnitude (important if values are negative)
dct = np.abs(dct)

# Log scaling (VERY IMPORTANT — same as your report)
dct = np.log(dct + 1)

# Normalize to 0–255
dct = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX)

# Convert to uint8
dct = dct.astype(np.uint8)

# Save image
cv2.imwrite("dct.png", dct)