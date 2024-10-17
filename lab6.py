import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_chain_code(direction):
    chain_code_dir = {
        (1, 0): 0,
        (1, -1): 1,
        (0, -1): 2,
        (-1, -1): 3,
        (-1, 0): 4,
        (-1, 1): 5,
        (0, 1): 6,
        (1, 1): 7
    }
    return chain_code_dir.get(direction, -1)

def calculate_chain_codes(contours):
    chain_codes = []
    for contour in contours:
        current_chain = []
        for i in range(len(contour)):
            dx = contour[(i + 1) % len(contour)][0][0] - contour[i][0][0]
            dy = contour[(i + 1) % len(contour)][0][1] - contour[i][0][1]
            chain_code = get_chain_code((dx, dy))
            current_chain.append(chain_code)
        chain_codes.append(current_chain)
    return chain_codes

# Încărcarea și verificarea imaginii binare
image_path = 'figuri.png'  # Ajustează calea conform configurației tale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Imaginea nu a putut fi încărcată de la calea {image_path}.")
    exit()

_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contoured_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

for i, contour in enumerate(contours):
    cv2.drawContours(contoured_image, [contour], -1, (0, 0, 255), 2)
    
    # Calculul centrului de greutate pentru etichetare
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Plasarea etichetei
        cv2.putText(contoured_image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow('Imaginea cu obiectele conturate', contoured_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

chain_codes = calculate_chain_codes(contours)

for i, code in enumerate(chain_codes):
    if i != 0:
        print(f"Contur {i}: Cod Înlănțuit = {code}")