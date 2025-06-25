import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/dragon.jpg", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
mean, std = 0, 20
noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Otsu's thresholding
_, otsu_thresh = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("../results/noisy_image.png", noisy_image)
cv2.imwrite("../results/otsu_result.png", otsu_thresh)

plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(otsu_thresh, cmap="gray")
plt.title("Otsu Result")
plt.axis("off")
plt.show()
