# %% [markdown]
#
# # CV Assignment 2
#
# Take two images. One should be your image. Second one should be an image of any historical place.
# With the help of segmentation, create an output image in which foreground will be your image and background will be the historical image.

# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import mediapipe as mp

# %%
image = cv.imread("../images/012.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
background_img = cv.imread("../images/adi-lica-ZpN1lhola0s-unsplash.jpg")
background_img = cv.cvtColor(background_img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.axis("off")
plt.imshow(image)

plt.subplot(122)
plt.axis("off")
plt.imshow(background_img)
plt.show()

# %%
pad_img = np.zeros(
    (image.shape[0], int(1.8 * image.shape[1]), 3), dtype=image.dtype
)
pad_img[:, pad_img.shape[1] - image.shape[1] :, :] = image
resized_bg_img = cv.resize(
    background_img, (pad_img.shape[1], pad_img.shape[0])
)

image = pad_img
bg_image = resized_bg_img

# %%
selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation
segment = selfie_seg(model_selection=0)
results = segment.process(image)

threshold = 0.5
binary_mask = results.segmentation_mask > threshold
mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
output_img = np.where(mask3d, image, bg_image)

plt.imshow(output_img)
plt.axis("off")
plt.show()
