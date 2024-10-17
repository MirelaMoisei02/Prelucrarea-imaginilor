import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math

def width_search(binary_image):
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.uint16)  # initializare matrice
    label_count = 0  # initializare numarare

    # defineste 8 pixeli vecini
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

    # face loop pt fiecare pixel din imag binara
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and labels[i, j] == 0:
                label_count += 1
                Q = deque()  # initializare coada BFS
                labels[i, j] = label_count
                Q.append((i, j))  

                while Q:
                    q = Q.popleft() 
                    for neighbor in N8(q):
                        ny, nx = neighbor
                        if binary_image[ny, nx] == 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label_count
                            Q.append(neighbor)

    return labels

def width_search_with_equivalence(binary_image):
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.uint16)  
    edges = [[] for _ in range(height * width)]  
    label_count = 0  

    # # defineste 8 pixeli vecini
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

    # primul traversat
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

    # al doilea traversat
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

    # actualizare
    for i in range(height):
        for j in range(width):
            labels[i, j] = newlabels[labels[i, j]]

    return labels

def calculate_contours(labels):
    contours = []
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.uint8(labels == label)
        contour, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours.append(contour[0])
    return contours

def calculate_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids

def calculate_diagonals(contours):
    diagonals = []
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        diagonal = math.sqrt(w**2 + h**2)
        diagonals.append(diagonal)
    return diagonals

def calculate_areas(contours):
    areas = []
    for contour in contours:
        area = cv.contourArea(contour)
        areas.append(area)
    return areas

def calculate_aspect_ratios(contours):
    aspect_ratios = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def calculate_convex_hulls(contours):
    convex_hulls = []
    for contour in contours:
        convex_hull = cv.convexHull(contour)
        convex_hulls.append(convex_hull)
    return convex_hulls

img = cv.imread("geo1.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 225, 255, cv.THRESH_BINARY)
labels = width_search(thresh)
labels_with_equivalence = width_search_with_equivalence(thresh)

contours = calculate_contours(labels_with_equivalence)
centroids = calculate_centroids(contours)
diagonals = calculate_diagonals(contours)
areas = calculate_areas(contours)
aspect_ratios = calculate_aspect_ratios(contours)
convex_hulls = calculate_convex_hulls(contours)

plt.figure(figsize=(18, 12))

plt.subplot(231), plt.imshow(img, cmap="gray"), plt.title("Imaginea Originala"), plt.axis("off")
plt.subplot(232), plt.imshow(thresh, cmap="gray"), plt.title("Imaginea Binara"), plt.axis("off")
plt.subplot(233), plt.imshow(labels_with_equivalence, cmap="nipy_spectral"), plt.title("Componente Conectate (Doua treceri cu clase de echivalenta)"), plt.axis("off")

for contour in contours:
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)

for centroid in centroids:
    plt.plot(centroid[0], centroid[1], 'bo')

for i, diagonal in enumerate(diagonals):
    plt.text(centroids[i][0], centroids[i][1], f"{diagonal:.2f}", fontsize=8, color='b')

plt.subplot(234), plt.imshow(labels_with_equivalence, cmap="nipy_spectral"), plt.title("Conturul"), plt.axis("off")
for contour in contours:
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)

plt.subplot(235), plt.imshow(labels_with_equivalence, cmap="nipy_spectral"), plt.title("Aria"), plt.axis("off")
for i, area in enumerate(areas):
    plt.text(centroids[i][0], centroids[i][1], f"{area:.2f}", fontsize=8, color='b')

plt.subplot(236), plt.imshow(labels_with_equivalence, cmap="nipy_spectral"), plt.title("Raport de aspect (L si l dintr-un dreptunghi)"), plt.axis("off")
for i, aspect_ratio in enumerate(aspect_ratios):
    plt.text(centroids[i][0], centroids[i][1], f"{aspect_ratio:.2f}", fontsize=8, color='b')

plt.tight_layout()
plt.show()