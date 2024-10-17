import numpy as np
import cv2
import matplotlib.pyplot as plt

def ideal_lowpass_filter(img, cutoff):
    # Transformata Fourier a imaginii
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # "trece jos"
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

def ideal_highpass_filter(img, cutoff):
    # Transformata Fourier discreta
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # "trece sus"
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)

    # Transformarea Fourier discreta inversa
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

img = cv2.imread('geo1.jpg', cv2.IMREAD_GRAYSCALE)

low_passed_img = ideal_lowpass_filter(img, 30)  # Ajustează cutoff-ul după nevoie
high_passed_img = ideal_highpass_filter(img, 30)  # Ajustează cutoff-ul după nevoie

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(low_passed_img, cmap='gray'), plt.title('Trece Jos')
plt.subplot(133), plt.imshow(high_passed_img, cmap='gray'), plt.title('Trece Sus')
plt.show()