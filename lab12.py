import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def adaptive_edge_binarization(image, threshold1, threshold2, ksize):
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges

def edge_extension(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# Citirea imaginii
img = cv2.imread('ghiocei.jpg', cv2.IMREAD_GRAYSCALE)

# Parametrii pentru binarizare adaptivă
threshold1 = 50
threshold2 = 150
ksize = 3

# Parametrii pentru prelungirea muchiilor
low_threshold = 30
high_threshold = 90

# Binarizare adaptivă a punctelor de muchie
start_adaptive_binarization_time = time.perf_counter()
adaptive_edges = adaptive_edge_binarization(img, threshold1, threshold2, ksize)
end_adaptive_binarization_time = time.perf_counter()
adaptive_binarization_time = (end_adaptive_binarization_time - start_adaptive_binarization_time) * 1000

# Prelungirea muchiilor prin histereză
start_edge_extension_time = time.perf_counter()
extended_edges = edge_extension(adaptive_edges, low_threshold, high_threshold)
end_edge_extension_time = time.perf_counter()
edge_extension_time = (end_edge_extension_time - start_edge_extension_time) * 1000

# Afișarea rezultatelor folosind matplotlib
plt.figure(figsize=(15, 5))

# Imaginea originala
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

# Muchiile binarizate adaptiv
plt.subplot(1, 3, 2)
plt.imshow(adaptive_edges, cmap='gray')
plt.title(f'Adaptive Edge Binarization\nTimp: {adaptive_binarization_time:.5f} sec')
plt.axis('off')

# Muchiile prelungite prin histereză
plt.subplot(1, 3, 3)
plt.imshow(extended_edges, cmap='gray')
plt.title(f'Edge Extension\nTimp: {edge_extension_time:.5f} sec')
plt.axis('off')

# Afișarea plotului
plt.tight_layout()
plt.show()