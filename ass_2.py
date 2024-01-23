import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# ------------------ Experiment 1 ----------------------------------------
# original image
img = cv2.imread('eight.jpg', 0)
img = img / 255

# show image
print(img.shape)
cv2.imshow('Eight Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian noise
x, y = img.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
noise = np.random.normal(loc=mean, scale=sigma, size=(x, y))


noisy_img = img + noise

cv2.imshow('Noisy image', noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#mean filter (average)

m = 3
n = 3
denoise_avg = cv2.blur(noisy_img, (m, n))

cv2.imshow('Denoise_Mean', denoise_avg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# median filter
img_noise_median = np.clip(noisy_img, -1, 1)  # pixel value range
img_noise_median = np.array(img_noise_median*255, dtype='uint8')
denoise_median = cv2.medianBlur(img_noise_median, 5)

cv2.imshow('Denoise_Median', denoise_median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------ Experiment 2 ----------------------------------------
# img2 = cv2.imread('circuit.jpg', 0)
# img2 = img2 / 255
#
# # Sobel_using_scipy
# dx, dy = ndimage.sobel(img2, axis=0), ndimage.sobel(img2, axis=1)
# sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
# sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step
#
# # Display and compare input and output images
# fig = plt.figure(1)
# ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
# ax1.imshow(img2, cmap='gray')
# ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
# plt.title("applying sobel kernel")
# plt.show()
#
#
# ###############################################################
#
# # laplacian (kernel 1)
# kernel = np.array([[0, 1, 0],
#                    [1, -4, 1],
#                    [0, 1, 0]])
#
# LaplacianImage = cv2.filter2D(src=img2, ddepth=-1, kernel=kernel)
#
# plt.figure(figsize=(8, 5))
# plt.imshow(LaplacianImage, cmap='gray')
# plt.title("Laplacian kernel")
# plt.axis('off')
# plt.show()
#
# c = 1
# g = img2 + c * LaplacianImage
#
# plt.figure(figsize=(8, 5), dpi=150)  #Dots per inches (dpi) determines how many pixels the figure comprises eg:6.4 inches * 100 dpi = 640 pixels
# plt.imshow(g, cmap='gray')
# plt.title("Laplacian Image")
# plt.axis('off')
# plt.show()
#
# gClip = np.clip(g, 0, 255)  # values smaller than 0-->0, and values larger than 1-->1
# plt.figure(figsize=(8, 5), dpi=150)
# plt.imshow(gClip, cmap='gray')
# plt.title("Circuit Image after applying")
# plt.axis('off')
# plt.show()
