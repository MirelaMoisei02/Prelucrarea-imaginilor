import cv2
import numpy as np
import sys
import os

# Citire imagini
img = cv2.imread('martisor.jpg')
img1 = cv2.imread('martisor_crenguta.jpg')
img2 = cv2.imread('buchet_flori_martisor.jpg')
img3 = cv2.imread('ghiocei.jpg')

# Afisare si salvare imagini in functie de alegerea utilizatorului
alegere = int(sys.argv[1])
if alegere == 1:

    # afisare imagine
    cv2.imshow('martisor',img)

    # salvare pe disk si confirmare daca imaginea a fost salvata
    cv2.imwrite('martisor_salvat.jpg', img)
    if os.path.exists('martisor_salvat.jpg'):
        print("Imaginea 'martisor' a fost salvată pe disc.")
    else:
        print("Eroare: Imaginea 'martisor' nu a fost salvată.")

elif alegere == 2:

    # afisare imagine
    cv2.imshow('martisor crenguta',img1)

    # salvare pe disk si confirmare daca imaginea a fost salvata
    cv2.imwrite('martisor_crenguta_salvat.jpg', img1)
    if os.path.exists('martisor_crenguta_salvat.jpg'):
        print("Imaginea 'martisor_crenguta' a fost salvată pe disc.")
    else:
        print("Eroare: Imaginea 'martisor_crenguta' nu a fost salvată.")

elif alegere == 3:
    # afisare imagine
    cv2.imshow('buchet flori martisor',img2)

    # salvare pe disk si confirmare daca imaginea a fost salvata
    cv2.imwrite('buchet_flori_martisor_salvat.jpg', img2)
    if os.path.exists('buchet_flori_martisor_salvat.jpg'):
        print("Imaginea 'buchet_flori_martisor' a fost salvată pe disc.")
    else:
        print("Eroare: Imaginea 'buchet_flori_martisor' nu a fost salvată.")
        
elif alegere == 4:
     # afisare imagine
    cv2.imshow('ghiocei',img3)

    # salvare pe disk si confirmare daca imaginea a fost salvata
    cv2.imwrite('ghiocei_salvat.jpg', img)
    if os.path.exists('ghiocei_salvat.jpg'):
        print("Imaginea 'ghiocei' a fost salvată pe disc.")
    else:
        print("Eroare: Imaginea 'ghiocei' nu a fost salvată.")
else:
    print("Nu exista")

cv2.waitKey(0)
cv2.destroyAllWindows()