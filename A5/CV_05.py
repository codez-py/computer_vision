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
# # SIFT

# %%
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

# %% [markdown]
# ## Loading Images

# %%
img1 = cv.imread('../images/train/Abdullah_Gul/Abdullah_Gul_0001.jpg')
img2 = cv.imread('../images/train/Abdullah_Gul/Abdullah_Gul_0003.jpg')
#img2 = cv.imread('../images/train/Ari_Fleischer/Ari_Fleischer_0001.jpg')
#img2 = cv.imread('../images/train/Alejandro_Toledo/Alejandro_Toledo_0001.jpg')
#img2 = cv.imread('../images/train/Amelie_Mauresmo/Amelie_Mauresmo_0001.jpg')

# %% [markdown]
# ## Face Detection

# %%
# for face detection
#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# images to grayscale
gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
gray2= cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

gray=[gray1,gray2]

# detect faces in the 2 images
faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
roi_gray=[]
roi_color=[]

size=gray1.shape

# crop out only the face of the first and second images
for (x,y,w,h) in faces1:

    extra=int(w/6)
    x1=max(0,x-extra)
    y1=max(0,y-extra)
    x2=min(size[1],x1+2*extra+w)
    y2=min(size[0],y1+2*extra+w)

    img1 = cv.rectangle(img1,(x1,y1),(x2-1,y2-1),(0,0,255),4)
    roi_gray .append(gray1[y1:y2, x1:x2])
    roi_color .append(img1[y1:y2, x1:x2])

if len(faces1)==0:
  roi_gray .append(gray1)
  roi_color .append(img1)
    
size=gray2.shape
for (x,y,w,h) in faces2:

    extra=int(w/6)
    x1=max(0,x-extra)
    y1=max(0,y-extra)
    x2=min(size[1],x1+2*extra+w)
    y2=min(size[0],y1+2*extra+w)

    img2 = cv.rectangle(img2,(x1,y1),(x2-1,y2-1),(0,0,255),4)
    roi_gray .append(gray2[y1:y2, x1:x2])
    roi_color .append(img2[y1:y2, x1:x2])

if len(faces2)==0:
  roi_gray .append(gray2)
  roi_color .append(img2)

# img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# roi_color=cv.cvtColor(roi_color,cv.COLOR_BGR2RGB)

plt.axis('off')
plt.imshow(roi_gray[0],cmap='gray')
plt.title('ROI of image 1')
plt.show()

plt.axis('off')
plt.imshow(roi_gray[1], cmap='gray')
plt.title('ROI of image 2')
plt.show()

# %% [markdown]
# ## Finding SIFT keypoints and descriptors

# %%
# using SIFT detect the feature descriptors of the 2 images
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(roi_gray[0],None)
kp2, des2 = sift.detectAndCompute(roi_gray[1],None)

# %% [markdown]
# ## Matching SIFT descriptors of 2 images

# %%
# Match descriptors.
bf = cv.BFMatcher(cv.NORM_L2)
matches=bf.knnMatch(des1, des2, k=2)

# %% [markdown]
# ## Finding Good matches

# %%
# Apply ratio test to filter out only the good matches
good = []
for m, n in matches:
    if m.distance < 0.85 * n.distance:
        good.append([m])

print('good matches', len(good))

# %%
img4 = cv.drawMatchesKnn(roi_gray[0],kp1,roi_gray[1],kp2,good,None,flags=2)
plt.axis('off')
plt.imshow(img4)
plt.show()


# %% [markdown]
# ## Training

# %%
def detect_faces(img1, img2):
    # for face detection
    #face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    # images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    gray2= cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    gray=[gray1,gray2]
    
    # detect faces in the 2 images
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    roi_gray=[]
    roi_color=[]
    
    size=gray1.shape
    
    # crop out only the face of the first and second images
    for (x,y,w,h) in faces1:
    
        extra=int(w/6)
        x1=max(0,x-extra)
        y1=max(0,y-extra)
        x2=min(size[1],x1+2*extra+w)
        y2=min(size[0],y1+2*extra+w)
    
        img1 = cv.rectangle(img1,(x1,y1),(x2-1,y2-1),(0,0,255),4)
        roi_gray .append(gray1[y1:y2, x1:x2])
        roi_color .append(img1[y1:y2, x1:x2])
    
    if len(faces1)==0:
      roi_gray .append(gray1)
      roi_color .append(img1)
        
    size=gray2.shape
    for (x,y,w,h) in faces2:
    
        extra=int(w/6)
        x1=max(0,x-extra)
        y1=max(0,y-extra)
        x2=min(size[1],x1+2*extra+w)
        y2=min(size[0],y1+2*extra+w)
    
        img2 = cv.rectangle(img2,(x1,y1),(x2-1,y2-1),(0,0,255),4)
        roi_gray .append(gray2[y1:y2, x1:x2])
        roi_color .append(img2[y1:y2, x1:x2])
    
    if len(faces2)==0:
      roi_gray .append(gray2)
      roi_color .append(img2)

    return roi_gray[0], roi_gray[1]


# %%
sift = cv.SIFT_create()
def feature_matching(img1, img2):
    img1, img2 = detect_faces(img1, img2)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    return len(matches)


# %%
folders = os.listdir('../images/train')
train_images = []
test_images = []
labels = []

for folder in folders:
    person = os.listdir('../images/train/' + folder)
    # first 10 images are taken for training
    for i in range(0, 10):
        img = cv.imread('../images/train/' + folder + '/' + person[i])
        train_images.append(img)
    # last 2 images are taken for testing
    for j in range(10, 12):
        img = cv.imread('../images/train/' + folder + '/' + person[j])
        test_images.append(img)
    labels.append(folder)

print(len(labels))
labels

# %%
featuremap = np.zeros((20, 100))
c = 0
#print(len(img2))

for (j, img2) in enumerate(test_images):
    l = []
    print(j)
    for img1 in train_images:
        l.append(feature_matching(img1, img2))
    featuremap[c] = l
    c = c + 1

# %%
featuremap

# %% [markdown]
# ## Testing

# %%
predicted = []
for i in range(featuremap.shape[0]):
    l = featuremap[i,1:]
    # find the index of maximum value
    ind = l.argmax() // 10 # 10 images in train
    matched = labels[ind]
    out = labels[i // 2] == matched
    predicted.append(out) # 2 images in test
    print(f"{out}, test:{labels[i//2]} is matched with train:{matched}")

# %%
print('accuracy:',  sum(predicted) / len(predicted))
