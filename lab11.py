import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Funcție nucleu Gaussian
def create_gaussian_kernel(ksize, sigma):
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    return kernel

# Funcție filtru Gaussian
def apply_gaussian_blur(image, ksize, sigma):
    kernel = create_gaussian_kernel(ksize, sigma)
    return cv2.filter2D(image, -1, kernel)

# Funcție filtru de medie (nucleu bidimensional)
def apply_mean_filter(image, ksize):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize ** 2)
    return cv2.filter2D(image, -1, kernel)

img = cv2.imread('ghiocei.jpg', cv2.IMREAD_GRAYSCALE)  

# Parametrii filtre
ksize = 5
sigma = 1

# Filtrare Gaussiana
start_gauss_time = time.perf_counter()
gaussian_blurred = apply_gaussian_blur(img, ksize, sigma)
end_gauss_time = time.perf_counter()
gaussian_time = (end_gauss_time - start_gauss_time) * 1000 

# Filtrare cu nucleu bidimensional de medie
start_mean_time = time.perf_counter()
mean_blurred = apply_mean_filter(img, ksize)
end_mean_time = time.perf_counter()
mean_time = (end_mean_time - start_mean_time) * 1000 


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_blurred, cmap='gray')
plt.title(f'Gaussian Blur\nTimp: {gaussian_time:.5f} sec')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mean_blurred, cmap='gray')
plt.title(f'Mean Filter\nTimp: {mean_time:.5f} sec')
plt.axis('off')

plt.tight_layout()
plt.show()