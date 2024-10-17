import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_thresholding(image):
    # Convertirea imaginii în imagine grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicarea binarizării globale cu metoda lui Otsu
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

def invert_image(image):
    return 255 - image

def adjust_contrast(image, alpha, beta):
    adjusted_image = np.clip(alpha * image + beta, 0, 255)
    return adjusted_image.astype(np.uint8)

def gamma_correction(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def adjust_brightness(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image



# Încărcarea imaginii
image_path = "geo1.jpg"
original_image = cv2.imread(image_path)

# Aplicarea operațiilor asupra imaginii
binary_image = global_thresholding(original_image)
inverted_image = invert_image(original_image)
contrast_adjusted_image = adjust_contrast(original_image, alpha=1.5, beta=10)
gamma_corrected_image = gamma_correction(original_image, gamma=1.5)
brightness_adjusted_image = adjust_brightness(original_image, value=50)
equalized_image = histogram_equalization(original_image)

# Afișarea imaginilor
'''cv2.imshow("Original Image", original_image)
cv2.imshow("Global Thresholding", binary_image)
cv2.imshow("Inverted Image", inverted_image)
cv2.imshow("Contrast Adjusted Image", contrast_adjusted_image)
cv2.imshow("Gamma Corrected Image", gamma_corrected_image)
cv2.imshow("Brightness Adjusted Image", brightness_adjusted_image)
cv2.imshow("Equalized Image", equalized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

# Afișarea imaginilor într-o figură
fig, axes = plt.subplots(2, 4, figsize=(20, 5))

axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Imaginea originala')

axes[0, 1].imshow(binary_image, cmap='gray')
axes[0, 1].set_title('Binarizare globala')

axes[0, 2].imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Imagine negativata')

axes[0, 3].imshow(cv2.cvtColor(contrast_adjusted_image, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('Modificarea contrastului')

axes[1, 0].imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Corectia Gamma')

axes[1, 1].imshow(cv2.cvtColor(brightness_adjusted_image, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Ajustarea luminozitatii')

axes[1, 2].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Histograma egalizata')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()