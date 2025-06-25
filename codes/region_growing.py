import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_growing(img, seed, threshold=30):
    output = np.zeros_like(img)
    visited = np.zeros_like(img, dtype=bool)
    h, w = img.shape
    to_visit = [seed]
    seed_val = img[seed]

    while to_visit:
        x, y = to_visit.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        if abs(int(img[x, y]) - int(seed_val)) <= threshold:
            output[x, y] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        to_visit.append((nx, ny))
    return output


image = cv2.imread("../images/dragon.jpg", cv2.IMREAD_GRAYSCALE)

# Seed point inside Object 1
seed_point = (55, 55)

region = region_growing(image, seed_point, threshold=30)

cv2.imwrite("../results/region_grown_result.png", region)

plt.imshow(region, cmap="gray")
plt.title("Region Growing Result")
plt.axis("off")
plt.show()
