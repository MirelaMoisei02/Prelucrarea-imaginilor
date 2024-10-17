import numpy as np
import cv2

# Imagine color
img_color = cv2.imread('martisor.jpg')

# Conversia la imagine in tonuri de gri
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Conversia la imagine binara
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Conversia din spatiul de culoare RGB în spațiul de culoare HSV
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# Afișarea imaginilor
cv2.imshow('Imagine color', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagine in tonuri de gri', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagine binara', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagine HSV', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()