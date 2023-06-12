# PRAKTIKUM 8 - EDGE DETECTION

import matplotlib.pyplot as plt
import cv2 as cv 
import numpy as np
from skimage.io import imread

#====================================
# FILTER SOBEL

img = cv.imread('LY.png',0)

img_sobelx = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)
img_sobely = cv.Sobel(img,cv.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_sobelx, cmap = 'gray')
ax[2].set_title("Sobel X")
ax[3].hist(img_sobelx.ravel(), bins = 256)
ax[3].set_title("Histogram Sobel x")

ax[4].imshow(img_sobely, cmap = 'gray')
ax[4].set_title("Sobel y")
ax[5].hist(img_sobely.ravel(), bins = 256)
ax[5].set_title("Histogram Sobel y")

ax[6].imshow(img_sobel, cmap = 'gray')
ax[6].set_title("Sobel xy")
ax[7].hist(img_sobel.ravel(), bins = 256)
ax[7].set_title("Histogram Sobel xy")

fig.tight_layout()
plt.show()

#====================================
# FILTER PREWITT

img = cv.imread("LY.png")

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv.filter2D(img, -1, kernelx)
img_prewitty = cv.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx, cmap = 'gray')
ax[2].set_title("Prewitt x")
ax[3].hist(img_prewittx.ravel(), bins = 256)
ax[3].set_title("Histogram Prewitt x")

ax[4].imshow(img_prewitty, cmap = 'gray')
ax[4].set_title("Prewitt y")
ax[5].hist(img_prewitty.ravel(), bins = 256)
ax[5].set_title("Histogram Prewitt y")

ax[6].imshow(img_prewitt, cmap = 'gray')
ax[6].set_title("Prewitt xy")
ax[7].hist(img_prewitt.ravel(), bins = 256)
ax[7].set_title("Histogram Prewitt xy")

fig.tight_layout()
plt.show()

#====================================
# FILTER CANNY

img = cv.imread("LY.png")

# Canny
img_canny = cv.Canny(img,100,200)

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny, cmap = 'gray')
ax[2].set_title("Citra Output")
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Output")

fig.tight_layout()
plt.show()

#====================================
# FILTER TURUNAN KEDUA

# Baca Gambar
img = cv.imread("LY.png")

# Konversi ke gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Hilangkan Noise
img1 = cv.GaussianBlur(gray,(3,3),0)

# Konvolusi dengan kernel
laplacian = cv.Laplacian(img,cv.CV_64F)

# Tampilkan dengan matplotlib
plt.subplot(1,2,1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
