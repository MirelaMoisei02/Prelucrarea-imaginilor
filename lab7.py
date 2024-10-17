import cv2
import numpy as np
import matplotlib.pyplot as plt

# Încărcarea imaginii în alb-negru
original_image = cv2.imread('geo1.jpg')
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Crearea unui element structurant
kernel = np.ones((5,5), np.uint8)

# Eroziunea imaginii
eroded_image = cv2.erode(image, kernel, iterations=1)

# Extragerea conturului
contour_image = cv2.subtract(image, eroded_image)

# Dilatarea pentru a închide spațiile mici
dilated_image = cv2.dilate(image, kernel, iterations=1)

# Eroziunea pentru a netezi contururile
filled_image = cv2.erode(dilated_image, kernel, iterations=1)

# Setarea figurii și axelor pentru afișarea ploturilor
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Afișarea imaginii originale
axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imaginea originala')
axs[0].axis('off')

# Afișarea imaginii cu contururi
axs[1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Contururi')
axs[1].axis('off')

# Afișarea imaginii cu regiunile umplute
axs[2].imshow(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
axs[2].set_title('Regiuni umplute')
axs[2].axis('off')

# Afișarea plotului
plt.show()