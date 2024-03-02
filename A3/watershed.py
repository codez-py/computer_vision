# %% [markdown]
# # Assignment 3: Watershed Algorithm

# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# %%
#input_file = "water_coins.jpg"
input_file = "coins.png"
img = cv.imread(input_file)
assert (
    img is not None
), "file could not be read, check with os.path.exists()"

print(img.shape)

# %%
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(
    gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)

plt.subplot(121)
plt.axis("off")
plt.imshow(img)
plt.title("Original Image")

plt.subplot(122)
plt.axis("off")
plt.imshow(thresh, cmap="gray")
plt.title("Binary image")

plt.show()

# %% [markdown]
# Find sure background region using dilation

# %%
### noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

plt.axis("off")
plt.imshow(sure_bg, cmap="gray")
plt.title("Sure Background")
plt.show()

# %% [markdown]
# Find sure foreground region using thresholded distance transform and erosion

# %%
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
ret, sure_fg = cv.threshold(
    # adjust this constant for better segmentation
    dist_transform, 0.5 * dist_transform.max(), 255, 0
)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

plt.subplot(121)
plt.axis("off")
plt.imshow(dist_transform, cmap="gray")
plt.title("Distance Transform")

plt.subplot(122)
plt.axis("off")
plt.imshow(sure_fg, cmap="gray")
plt.title("Thresholded distance transform, sure FG")

plt.show()

# %% [markdown]
# Marking each connected component as one segment

# %%
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

plt.axis("off")
plt.imshow(markers)
plt.title("Image with Marked Sure Foreground")
plt.show()

# %% [markdown]
# Apply watershed algorithm with labelled sure foreground as minima, we can
# find labels for unknown regions and which segment it belongs

# %%
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

plt.subplot(121)
plt.axis("off")
plt.imshow(markers)
plt.title("Segmented Marker Image")

plt.subplot(122)
plt.axis("off")
plt.imshow(img)
plt.title("Segmented Image")

plt.show()

# %%
print("No of segments:", markers.max())
# 1 segment will be the background
print("No of coins:", markers.max() - 1)
