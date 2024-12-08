import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(15, 6))
plt.suptitle("Grayscale histograms", fontsize=14, fontweight='bold')
image = cv2.imread('images/lenna.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 4, 'm')

M = np.ones(grayImage.shape, dtype="uint8") * 35
addedImage = cv2.add(grayImage, M)
histAddedImage = cv2.calcHist([addedImage], [0], None, [256], [0, 256])
subtractedImage = cv2.subtract(grayImage, M)
histSubtractedImage = cv2.calcHist([subtractedImage], [0], None, [256], [0, 256])

ShowImgWithMatplotlib(cv2.cvtColor(addedImage, cv2.COLOR_GRAY2BGR), "gray lighter", 2)
ShowHistWithMatplotlibGray(histAddedImage, "grayscale histogram", 5, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(subtractedImage, cv2.COLOR_GRAY2BGR), "gray darker", 3)
ShowHistWithMatplotlibGray(histSubtractedImage, "grayscale histogram", 6, 'm')

plt.show()