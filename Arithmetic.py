import numpy as np
import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Arithmetic with images", fontsize=14, fontweight='bold')

image = cv2.imread('images/lenna.png')
M = np.ones(image.shape, dtype="uint8") * 60
addedImage = cv2.add(image, M)
subtractedImage = cv2.subtract(image, M)

scalar = np.ones((1, 3), dtype="float") * 110
addedImage2 = cv2.add(image, scalar)
subtractedImage2 = cv2.subtract(image, scalar)

ShowWithMatplotlib(image, "image", 1)
ShowWithMatplotlib(addedImage, "added 60 (image + image)", 2)
ShowWithMatplotlib(subtractedImage, "subtracted 60 (image - images)", 3)
ShowWithMatplotlib(addedImage2, "added 110 (image + scalar)", 5)
ShowWithMatplotlib(subtractedImage2, "subtracted 110 (image - scalar)", 6)
plt.show()