import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Experiment 1 ----------------------------------------
# load image
img = cv2.imread('c.jpg', cv2.IMREAD_GRAYSCALE)

# show image
print(img.shape)
plt.imshow(img, cmap='gray')
plt.title("image in grayscale")

# calculate image normal histogram
hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure()
plt.plot(hist)
plt.title("image normal histogram")

# histogram shift by value
modified_img = cv2.add(img, 50)
plt.figure()
plt.imshow(modified_img, cmap='gray')
plt.title("image shifted by value")

# histogram of image shifted by value
modified_hist = cv2.calcHist([modified_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure()
plt.plot(modified_hist)
plt.title("image shifted histogram")

# calculate image cumulative histogram
cumulative_hist = hist.cumsum()
plt.figure()
plt.plot(cumulative_hist)
plt.title("image cumulative histogram")

# perform histogram equalization
modified_img = cv2.equalizeHist(img)
plt.figure()
plt.imshow(modified_img, cmap='gray')
plt.title("image after perform histogram equalization")

# calculate image modified histogram
modified_hist = cv2.calcHist([modified_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure()
plt.plot(modified_hist)
plt.title("image modified histogram")

# calculate image cumulative histogram
modified_cumulative_hist = modified_hist.cumsum()
plt.figure()
plt.plot(modified_cumulative_hist)
plt.title("image modified cumulative histogram")

# -------------------- Experiment 2 ----------------------------------------
# load image
img = cv2.imread('c.jpg', cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

r1, r2 = 0, 256

for i in range(0, hist.shape[0]):
    if hist[i][0] != 0:
        r1 = i
for i in range(hist.shape[0]-1, 0, -1):
    if hist[i][0] != 0:
        r2 = i

contrast_ratio = (256-0)/(r2-r1)
lut = np.array([(x-r1)*contrast_ratio for x in range(0, 256)], np.uint8)
modified_img = cv2.LUT(img, lut)

plt.figure()
plt.imshow(modified_img, cmap='gray')
plt.title("image LUT")

# show all plots
plt.show()
