import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def width_search(binary_image):
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.uint16)  # Initialize labels matrix
    label_count = 0  # Initialize label counter

    # Define 8-neighborhood for a pixel
    def N8(point):
        y, x = point
        neighbors = [
            (y - 1, x - 1),
            (y - 1, x),
            (y - 1, x + 1),
            (y, x - 1),
            (y, x + 1),
            (y + 1, x - 1),
            (y + 1, x),
            (y + 1, x + 1),
        ]
        return filter(lambda p: 0 <= p[0] < height and 0 <= p[1] < width, neighbors)

    # Loop through each pixel in the binary image
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and labels[i, j] == 0:
                label_count += 1
                Q = deque()  # Initialize queue for BFS
                labels[i, j] = label_count
                Q.append((i, j))  # Push current pixel to queue

                while Q:
                    q = Q.popleft()  # Pop the front element from the queue
                    for neighbor in N8(q):
                        ny, nx = neighbor
                        if binary_image[ny, nx] == 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label_count
                            Q.append(neighbor)

    return labels


def width_search_with_equivalence(binary_image):
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.uint16)  # Initialize labels matrix
    edges = [[] for _ in range(height * width)]  # Initialize edges list of lists
    label_count = 0  # Initialize label counter

    # Define 8-neighborhood for a pixel
    def N8(point):
        y, x = point
        neighbors = [
            (y - 1, x - 1),
            (y - 1, x),
            (y - 1, x + 1),
            (y, x - 1),
            (y, x + 1),
            (y + 1, x - 1),
            (y + 1, x),
            (y + 1, x + 1),
        ]
        return filter(lambda p: 0 <= p[0] < height and 0 <= p[1] < width, neighbors)

    # First traversal: Label connected components and build edges
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and labels[i, j] == 0:
                L = []
                for neighbor in N8((i, j)):
                    ny, nx = neighbor
                    if labels[ny, nx] > 0:
                        L.append(labels[ny, nx])
                if len(L) == 0:
                    label_count += 1
                    labels[i, j] = label_count
                else:
                    x = min(L)
                    labels[i, j] = x
                    for y in L:
                        if y != x:
                            edges[x].append(y)
                            edges[y].append(x)

    # Second traversal: Assign new labels using equivalence classes
    newlabel = 0
    newlabels = np.zeros(label_count + 1, dtype=np.uint16)
    for i in range(1, label_count + 1):
        if newlabels[i] == 0:
            newlabel += 1
            Q = deque()
            newlabels[i] = newlabel
            Q.append(i)
            while Q:
                x = Q.popleft()
                for y in edges[x]:
                    if newlabels[y] == 0:
                        newlabels[y] = newlabel
                        Q.append(y)

    # Update labels with new labels from equivalence classes
    for i in range(height):
        for j in range(width):
            labels[i, j] = newlabels[labels[i, j]]

    return labels


# Read the image and convert it to grayscale
img = cv.imread("geo.jpg")

# Conversia la imagine in tonuri de gri
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
ret, thresh = cv.threshold(img_gray, 225, 255, cv.THRESH_BINARY)

# Perform width search algorithm to find connected components
labels = width_search(thresh)

# Perform width search algorithm with equivalence classes
labels_with_equivalence = width_search_with_equivalence(thresh)

# Display the original image, binary image, labeled connected components without and with equivalence classes
plt.figure(figsize=(10, 5))

plt.subplot(221), plt.imshow(img, cmap="gray"), plt.title(
    "Imaginea Originala"
), plt.axis("off")
plt.subplot(222), plt.imshow(thresh, cmap="gray"), plt.title(
    "Imaginea Binara"
), plt.axis("off")
plt.subplot(223), plt.imshow(labels, cmap="nipy_spectral"), plt.title(
    "Componente Conectate (Traversare in latime)"
), plt.axis("off")
plt.subplot(224), plt.imshow(labels_with_equivalence, cmap="nipy_spectral"), plt.title(
    "Componente Conectate (Doua treceri cu clase de echivalenta)"
), plt.axis("off")

plt.tight_layout()
plt.show()