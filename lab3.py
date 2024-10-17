import numpy as np
import cv2
import matplotlib.pyplot as plt

# Imagine color
img_color = cv2.imread('martisor.jpg')

# Conversia la imagine in tonuri de gri
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Afișarea imaginii color
cv2.imshow('Imagine color', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Afișarea imaginii in tonuri de gri
cv2.imshow('Imagine in tonuri de gri', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculul histogramelor pentru fiecare canal de culoare
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img_color], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])

plt.title('Histograma imaginii color')
plt.show()

# Histograma imaginii in tonuri de gri
plt.hist(img_gray.ravel(), 256, [0,256])
plt.title('Histograma imaginii în tonuri de gri')
plt.show()

# Definirea pragurilor
thresholds = [50, 100, 150]  # Poți ajusta aceste praguri în funcție de necesități

# Aplicarea pragurilor
thresholded_images = [cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY)[1] for t in thresholds]

# Afișarea imaginilor rezultate
for i, thresholded_img in enumerate(thresholded_images):
    cv2.imshow(f'Imagine prag {thresholds[i]}', thresholded_img)

cv2.waitKey(0)
cv2.destroyAllWindows()



# Algoritmul de corecție Floyd-Steinberg
Height = img_gray.shape[0]
Width = img_gray.shape[1]

for y in range(0, Height):
    for x in range(0, Width):

        old_value = img_gray[y, x]
        new_value = 0
        if (old_value > 128):
            new_value = 255

        img_gray[y, x] = new_value

        Error = old_value - new_value

        if (x < Width - 1):
            NewNumber = img_gray[y, x + 1] + Error * 7 / 16
            if (NewNumber > 255): NewNumber = 255
            elif (NewNumber < 0): NewNumber = 0
            img_gray[y, x + 1] = NewNumber

        if (x > 0 and y < Height - 1):
            NewNumber = img_gray[y + 1, x - 1] + Error * 3 / 16
            if (NewNumber > 255): NewNumber = 255
            elif (NewNumber < 0): NewNumber = 0
            img_gray[y + 1, x - 1] = NewNumber

        if (y < Height - 1):
            NewNumber = img_gray[y + 1, x] + Error * 5 / 16
            if (NewNumber > 255): NewNumber = 255
            elif (NewNumber < 0): NewNumber = 0
            img_gray[y + 1, x] = NewNumber

        if (y < Height - 1 and x < Width - 1):
            NewNumber = img_gray[y + 1, x + 1] + Error * 1 / 16
            if (NewNumber > 255): NewNumber = 255
            elif (NewNumber < 0): NewNumber = 0
            img_gray[y + 1, x + 1] = NewNumber

# Afișarea imaginii rezultate
cv2.imshow('Imagine cu dithering folosind algoritmul Floyd-Steinberg', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()