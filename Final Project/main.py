import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def low_pass_filter(image, rad):
    im = np.array(image)
    im_shape = im.shape
    filter = np.zeros(im_shape)
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    mask = np.sqrt((x - im_shape[0] / 2) ** 2 + (y - im_shape[1] / 2) ** 2) <= rad
    filter[mask] = 1
    im = filter * im
    return im


def high_pass_filter(image, rad):
    im = np.array(image)
    im_shape = im.shape
    filter = np.zeros(im_shape)
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    mask = np.sqrt((x - im_shape[0] / 2) ** 2 + (y - im_shape[1] / 2) ** 2) >= rad
    filter[mask] = 1
    im = filter * im
    return im


def gaussian_low_pass_filter(image, rad):
    im = np.array(image)
    im_shape = im.shape
    gaussian_filter = np.zeros(im_shape)
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    gaussian_filter[:, :] = np.exp(-(np.sqrt((x - im_shape[0] / 2) ** 2 + (y - im_shape[1] / 2) ** 2)) / (2 * rad**2))
    im = gaussian_filter * im
    return im


def gaussian_high_pass_filter(image, rad):
    im = np.array(image)
    im_shape = im.shape
    gaussian_filter = np.zeros(im_shape)
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    gaussian_filter[:, :] = 1 - np.exp(-(np.sqrt((x - im_shape[0] / 2) ** 2 + (y - im_shape[1] / 2) ** 2)) / (2 * rad**2))
    im = gaussian_filter * im
    return im


img = plt.imread("./images_gray/Cars0.png")
# Define the sigma value for the Gaussian filter
sigma = 3
# Apply the Gaussian filter to the image
gaussian_filtered = gaussian_low_pass_filter(img, sigma).astype(np.float32)
# Compute the high-pass filtered image by subtracting the Gaussian-filtered image from the original image
high_pass_filtered = img - gaussian_filtered

# Display the original image, the Gaussian-filtered image, and the high-pass filtered image
fig, ax = plt.subplots(1, 5, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(gaussian_filtered, cmap='gray')
ax[1].set_title('Gaussian-filtered Image')
ax[2].imshow(high_pass_filtered, cmap='gray')
ax[2].set_title('High-pass filtered Image')
# threshold using cv
high_pass_filtered = (high_pass_filtered*255).astype(np.uint8)
thresh = cv2.adaptiveThreshold(high_pass_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=21, C=20)
thresh2 = cv2.threshold(high_pass_filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Display the original and threshold images

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=1)
# thresh = cv2.bitwise_xor(thresh, thresh1)
thresh = (thresh*255).astype(np.uint8)
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
img *= 0
img += 255
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)

ax[3].imshow(thresh, cmap='gray') # 120 80 160 40
ax[3].set_title('thresh Image')
ax[4].imshow(img, cmap='gray') # 120 80 160 40
ax[4].set_title('cnts Image')
plt.show()
