# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# +
img = cv.imread('012.jpg')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

plt.subplot(121)
plt.imshow(rgb_img)
plt.title('RGB Image')

plt.subplot(122)
plt.imshow(grayscale_img, cmap = 'gray')
plt.title('Grayscale Image')

plt.show()
# -

# # 1. Canny Edge Detection Algorithm
#
# Input: Image f ; Output: Edge
# Image E
# 1. Find g = Conv (f , h), where h
# is the Gaussian Filter
# (Smoothing Step)
# 2. Find Gradient of g, say g
# 3. Find Magnitude M and
# Orientation θ of g
# 4. Find S(x, y ) = 0 if
# M(x, y ) < M(x′, y′) for some
# (x′, y ′) in the direction of θ ;
# S(x, y ) = M(x, y ) otherwise
# (Non-Maximal Suppression
# Step)
# 5. E (x, y ) = 0 if S(x, y ) < T0
# e(x, y ) = 1 if S(x, y ) > T1
# (Double Thresholding)
# 6. E (x, y ) = 1 if E (x, y ) is not
# yet defined, and E (x′, y ′) = 1,
# where (x′, y ′) is adjacent to
# (x, y )
# (Edge tracing step)

# +
c = 60

canny_edges = cv.Canny(grayscale_img, 15, 30)
sharpened_img = grayscale_img + c * canny_edges

plt.subplot(131)
plt.imshow(grayscale_img, cmap = 'gray')
plt.title('Grayscale Image')

plt.subplot(132)
plt.imshow(canny_edges, cmap = 'gray')
plt.title('Canny Edge Image')

plt.subplot(133)
plt.imshow(sharpened_img, cmap = 'gray')
plt.title('Sharpened Image')

plt.show()
# -

# # 2. Marr Hildreth Edge Detection Algorithm
#
# Input: Image f ; Output: Edge Images E(x,y. σ)
# 1. Find g = Conv (f , h), where h is the Gaussian Filter with SD σ
# (Smoothing Step)
# 2. Find Laplacian of g, say ∆2g
# 3. Define E (x, y , σ) = 1 if Zero crossing occurs at (x, y ) at the scale σ
# E (x, y , σ) = 0 otherwise

# +
# 1. Gaussian
sigma = 4.456
gaussian_img = cv.GaussianBlur(grayscale_img, (0, 0), sigma)

# 2. Laplacian
laplacian = cv.Laplacian(gaussian_img, cv.CV_64F)

# 3. Zero crossing
signed_laplacian = np.sign(laplacian)
padded_laplacian = np.pad(signed_laplacian, ((0, 1), (0, 1)))
diff_x = padded_laplacian[:-1, :-1] - padded_laplacian[:-1, 1:] < 0
diff_y = padded_laplacian[:-1, :-1] - padded_laplacian[1:, :-1] < 0
mh_edges =  np.logical_or(diff_x, diff_y).astype(float)

c = 50
sharpened_img = grayscale_img + c * mh_edges

plt.subplot(131)
plt.imshow(grayscale_img, cmap = 'gray')
plt.title('Grayscale Image')

plt.subplot(132)
plt.imshow(mh_edges, cmap = 'gray')
plt.title('Marr Hildreth Edges')

plt.subplot(133)
plt.imshow(sharpened_img, cmap = 'gray')
plt.title('Sharpened Image')

plt.show()
