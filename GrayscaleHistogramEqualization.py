import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax = plt.subplot(3, 4, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(18, 14))
plt.suptitle("Grayscale histogram equalization with cv2.equalizeHist()", fontsize=16, fontweight='bold')

image = cv2.imread('images/lenna.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])
grayImageEq = cv2.equalizeHist(grayImage)
histEq = cv2.calcHist([grayImageEq], [0], None, [256], [0, 256])

M = np.ones(grayImage.shape, dtype="uint8") * 35
addedImage = cv2.add(grayImage, M)
histAddedImage = cv2.calcHist([addedImage], [0], None, [256], [0, 256])
addedImageEq = cv2.equalizeHist(addedImage)
histEqAddedImage = cv2.calcHist([addedImageEq], [0], None, [256], [0, 256])
subtractedImage = cv2.subtract(grayImage, M)
histSubtractedImage = cv2.calcHist([subtractedImage], [0], None, [256], [0, 256])
subtractedImageEq = cv2.equalizeHist(subtractedImage)
histEqSubtractedImage = cv2.calcHist([subtractedImageEq], [0], None, [256], [0, 256])

ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 2, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(addedImage, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
ShowHistWithMatplotlibGray(histAddedImage, "grayscale histogram", 6, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(subtractedImage, cv2.COLOR_GRAY2BGR), "gray darker", 9)
ShowHistWithMatplotlibGray(histSubtractedImage, "grayscale histogram", 10, 'm')

ShowImgWithMatplotlib(cv2.cvtColor(grayImageEq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3)
ShowHistWithMatplotlibGray(histEq, "grayscale equalized histogram", 4, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(addedImageEq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
ShowHistWithMatplotlibGray(histEqAddedImage, "grayscale equalized histogram", 8, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(subtractedImageEq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
ShowHistWithMatplotlibGray(histEqSubtractedImage, "grayscale equalized histogram", 12, 'm')

plt.show()