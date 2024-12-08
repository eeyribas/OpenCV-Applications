import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibRgb(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

def HistColorImg(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr

plt.figure(figsize=(15, 6))
plt.suptitle("Color histograms", fontsize=14, fontweight='bold')

image = cv2.imread('images/lenna.png')
histColor = HistColorImg(image)
ShowImgWithMatplotlib(image, "image", 1)
ShowHistWithMatplotlibRgb(histColor, "color histogram", 4, ['b', 'g', 'r'])

M = np.ones(image.shape, dtype="uint8") * 15
addedImage = cv2.add(image, M)
histColorAddedImage = HistColorImg(addedImage)
subtractedImage = cv2.subtract(image, M)
histColorSubtractedImage = HistColorImg(subtractedImage)

ShowImgWithMatplotlib(addedImage, "image lighter", 2)
ShowHistWithMatplotlibRgb(histColorAddedImage, "color histogram", 5, ['b', 'g', 'r'])
ShowImgWithMatplotlib(subtractedImage, "image darker", 3)
ShowHistWithMatplotlibRgb(histColorSubtractedImage, "color histogram", 6, ['b', 'g', 'r'])

plt.show()