import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create blank image
image = np.zeros((200, 200), dtype=np.uint8)

# Add Object 1
image[50:100, 50:100] = 100

# Add Object 2
image[120:170, 120:170] = 200

# Save image
cv2.imwrite("../images/original_image.png", image)
print("Synthetic image saved as images/original_image.png")

# Optional display
plt.imshow(image, cmap="gray")
plt.title("Synthetic Image")
plt.axis("off")
plt.show()
