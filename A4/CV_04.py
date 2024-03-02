# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment 4 - Bokeh Effect

# %%
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

plt.rcParams["figure.figsize"]= (10,10)
np.set_printoptions(precision=3)

# %% [markdown]
# ## Loading the Image

# %%
image = cv2.imread('../images/014.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.show()

# %%
triangle = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype='float')

mask = triangle
kernel = cv2.getGaussianKernel(11, 5.)
kernel = kernel * kernel.transpose() * mask # Is the 2D filter
kernel = kernel / np.sum(kernel)
print(kernel)


# %% [markdown]
# ## Bokeh Effect

# %%
def bokeh(image):
    r,g,b = cv2.split(image)

    r = r / 255.
    g = g / 255.
    b = b / 255.

    r = np.where(r > 0.9, r * 2, r)
    g = np.where(g > 0.9, g * 2, g)
    b = np.where(b > 0.9, b * 2, b)

    fr = cv2.filter2D(r, -1, kernel)
    fg = cv2.filter2D(g, -1, kernel)
    fb = cv2.filter2D(b, -1, kernel)

    fr = np.where(fr > 1., 1., fr)
    fg = np.where(fg > 1., 1., fg)
    fb = np.where(fb > 1., 1., fb)

    result = cv2.merge((fr, fg, fb))
    return result

result = bokeh(image)
plt.imshow(result)
plt.axis('off')
plt.show()
# face is also blurred in this image

# %% [markdown]
# ## Segmenting Image before applying Bokeh Effect

# %%
selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation
segment = selfie_seg(model_selection=0)
results = segment.process(image)

threshold = 0.5
binary_mask = results.segmentation_mask > threshold
mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
subject = np.where(mask3d, image, 0)
background = np.where(mask3d, 0, image)

plt.subplot(121)
plt.imshow(subject)
plt.title('Foreground')
plt.axis("off")

plt.subplot(122)
plt.imshow(background)
plt.axis("off")
plt.title('Background')
plt.show()

# %% [markdown]
# ## Applying Bokeh for background and combining with foreground image

# %%
background_bokeh = bokeh(np.asarray(background, dtype='uint8'))
background_bokeh = np.asarray(background_bokeh * 255, dtype='uint8')
combined = cv2.addWeighted(subject, 1., background_bokeh, 1., 0)
plt.imshow(combined)
plt.title('Image with Bokeh Effect')
plt.axis('off')
plt.show()

# %%
plt.imshow(image)
plt.title('Original Image')
plt.axis("off")
plt.show()

# %% [markdown]
# ## Bokeh Effect For another image

# %%
image = cv2.imread('../images/selfie_original.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation
segment = selfie_seg(model_selection=0)
results = segment.process(image)

threshold = 0.5
binary_mask = results.segmentation_mask > threshold
mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
subject = np.where(mask3d, image, 0)
background = np.where(mask3d, 0, image)

plt.subplot(121)
plt.imshow(subject)
plt.title('Foreground')
plt.axis("off")

plt.subplot(122)
plt.imshow(background)
plt.axis("off")
plt.title('Background')
plt.show()

# %%
background_bokeh = bokeh(np.asarray(background, dtype='uint8'))
background_bokeh = np.asarray(background_bokeh * 255, dtype='uint8')
combined = cv2.addWeighted(subject, 1.1, background_bokeh, 0.9, 0)

plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(combined)
plt.title('Image with Bokeh Effect')
plt.axis('off')
plt.show()
