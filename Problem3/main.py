import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy import signal, ndimage
from skimage import color, data, restoration

# ------------------ Experiment 1 ----------------------------------------
# original image
f = cv2.imread('cameraman.jpg', 0)

plt.imshow(f, cmap='gray')
plt.title("Original Image (cameraman)")
plt.axis('off')
plt.show()

# image in frequency domain
F = np.fft.fft2(f)
# plt.imshow(np.log1p(np.abs(F)), cmap='gray')
# plt.axis('off')
# plt.show()

Fshift = np.fft.fftshift(F)
# plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
# plt.axis('off')
# plt.show()

# Filter: Low pass filter
M, N = f.shape
H = np.zeros((M, N), dtype=np.float32)
D0 = 50
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        if D <= D0:
            H[u, v] = 1
        else:
            H[u, v] = 0

# plt.imshow(H, cmap='gray')
# plt.axis('off')
# plt.show()

# Ideal Low Pass Filtering
Gshift = Fshift * H
# plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
# plt.axis('off')
# plt.show()

# Inverse Fourier Transform
G = np.fft.ifftshift(Gshift)
# plt.imshow(np.log1p(np.abs(G)), cmap='gray')
# plt.axis('off')
# plt.show()

g = np.abs(np.fft.ifft2(G))
plt.imshow(g, cmap='gray')
plt.title("applying Low Pass filter")
plt.axis('off')
plt.show()

# Filter: High pass filter
H = 1 - H

# plt.imshow(H, cmap='gray')
# plt.axis('off')
# plt.show()

# Ideal High Pass Filtering
Gshift = Fshift * H
# plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
# plt.axis('off')
# plt.show()

# Inverse Fourier Transform
G = np.fft.ifftshift(Gshift)
# plt.imshow(np.log1p(np.abs(G)), cmap='gray')
# plt.axis('off')
# plt.show()

g = np.abs(np.fft.ifft2(G))
plt.imshow(g, cmap='gray')
plt.title("applying High Pass filter")
plt.axis('off')
plt.show()

# #----------------- Experiment 2 ----------------------------------------

f2 = cv2.imread('eight.jpg', 0)

plt.figure(figsize=(5, 5))
plt.imshow(f2, cmap='gray')
plt.title("Original Image (eight)")
plt.axis('off')
plt.show()

noisy_img = f2 + 20 * np.random.normal(size=f2.shape)
plt.figure(figsize=(5, 5))
plt.imshow(noisy_img, cmap='gray')
plt.title("Guassian_img")
plt.axis('off')
plt.show()

# Apply Adaptive Wiener Filter
filtered_img = wiener(noisy_img, (5, 5))
plt.figure(figsize=(5, 5))
plt.imshow(filtered_img, cmap='gray')
plt.title("Wiener")
plt.axis('off')
plt.show()

# # ------------------ Experiment 3 ----------------------------------------
# Salt and Pepper Noise
x, y = f.shape
g = np.zeros((x, y), dtype=np.float32)
pepper = 0.1
salt = 0.95
for i in range(x):
    for j in range(y):
        rdn = np.random.random()
        if rdn < pepper:
            g[i][j] = 0
        elif rdn > salt:
            g[i][j] = 1
        else:
            g[i][j] = f[i][j]

img_noise = g

plt.figure(figsize=(5, 5))
plt.imshow(img_noise, cmap='gray')
plt.title("applying salt and pepper noise")
plt.axis('off')
plt.show()

# mean filter (average)
m = 3
n = 3
denoise_avg = cv2.blur(img_noise, (m, n))
plt.figure(figsize=(5, 5))
plt.imshow(denoise_avg, cmap='gray')
plt.title("applying mean filter")
plt.axis('off')
plt.show()

# median filter
filter_size = 7
denoise_median = ndimage.median_filter(img_noise, size=filter_size)
plt.figure(figsize=(5, 5))
plt.imshow(denoise_median, cmap='gray')
plt.title("applying median filter")
plt.axis('off')
plt.show()


ksize = 3
# Apply the min filter
#min_img = cv2.erode(img_noise, np.ones((ksize, ksize), np.uint8))
min_img = ndimage.minimum_filter(img_noise, size=ksize)

plt.figure(figsize=(5, 5))
plt.imshow(min_img, cmap='gray')
plt.title("applying min filter")
plt.axis('off')
plt.show()

# Apply the max filter
#max_img = cv2.dilate(img_noise, np.ones((ksize, ksize), np.uint8))
max_img = ndimage.maximum_filter(img_noise, size=ksize)

plt.figure(figsize=(5, 5))
plt.imshow(max_img, cmap='gray')
plt.title("applying max filter")
plt.axis('off')
plt.show()
