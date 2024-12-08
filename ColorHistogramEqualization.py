import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibRgb(hist, title, pos, color):
    ax = plt.subplot(3, 4, pos)
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

def EqualizeHistColor(img):
    channels = cv2.split(img)
    eqChannels = []
    for ch in channels:
        eqChannels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eqChannels)
    return eq_image

plt.figure(figsize=(18, 14))
plt.suptitle("Color histogram equalization with cv2.equalizeHist() - not a good approach", fontsize=14, fontweight='bold')

image = cv2.imread('images/lenna.png')
histColor = HistColorImg(image)
imageEq = EqualizeHistColor(image)
histImageEq = HistColorImg(imageEq)

M = np.ones(image.shape, dtype="uint8") * 15
addedImage = cv2.add(image, M)
histColorAddedImage = HistColorImg(addedImage)
addedImageEq = EqualizeHistColor(addedImage)
histAddedImageEq = HistColorImg(addedImageEq)
subtractedImage = cv2.subtract(image, M)
histColorSubtractedImage = HistColorImg(subtractedImage)
subtractedImageEq = EqualizeHistColor(subtractedImage)
histSubtractedImageEq = HistColorImg(subtractedImageEq)

ShowImgWithMatplotlib(image, "image", 1)
ShowHistWithMatplotlibRgb(histColor, "color histogram", 2, ['b', 'g', 'r'])
ShowImgWithMatplotlib(addedImage, "image lighter", 5)
ShowHistWithMatplotlibRgb(histColorAddedImage, "color histogram", 6, ['b', 'g', 'r'])
ShowImgWithMatplotlib(subtractedImage, "image darker", 9)
ShowHistWithMatplotlibRgb(histColorSubtractedImage, "color histogram", 10, ['b', 'g', 'r'])

ShowImgWithMatplotlib(imageEq, "image equalized", 3)
ShowHistWithMatplotlibRgb(histImageEq, "color histogram equalized", 4, ['b', 'g', 'r'])
ShowImgWithMatplotlib(addedImageEq, "image lighter equalized", 7)
ShowHistWithMatplotlibRgb(histAddedImageEq, "color histogram equalized", 8, ['b', 'g', 'r'])
ShowImgWithMatplotlib(subtractedImageEq, "image darker equalized", 11)
ShowHistWithMatplotlibRgb(histSubtractedImageEq, "color histogram equalized", 12, ['b', 'g', 'r'])

plt.show()