import cv2
import numpy as np

# Definirea funcției pentru filtrul "trece jos"
def low_pass_filter(image, kernel_size=3):
    # Definirea kernelului de convoluție (poate fi de dimensiune variabilă)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # Aplicarea filtrului "trece jos" folosind funcția de convoluție
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Definirea funcției pentru filtrul "trece sus"
def high_pass_filter(image):
    # Aplicarea filtrului Laplacian pentru evidențierea muchiilor
    filtered_image = cv2.Laplacian(image, cv2.CV_64F)
    return filtered_image

# Încărcarea imaginii
image1 = cv2.imread('martisor.jpg')
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


# Aplicarea filtrului "trece jos"
smoothed_image = low_pass_filter(image)

# Aplicarea filtrului "trece sus"
edges_image = high_pass_filter(image)

# Afișarea imaginilor
cv2.imshow('Imaginea originala', image)
cv2.imshow('Trece jos', smoothed_image)
cv2.imshow('Trece sus', edges_image)
cv2.waitKey(0)
cv2.destroyAllWindows()