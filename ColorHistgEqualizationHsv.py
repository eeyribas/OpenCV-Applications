import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(3, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibRgb(hist, title, pos, color):
    ax=plt.subplot(3, 4, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

def HistColorImg(img):
    histr=[]
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr

def EqualizeHistColorHsv(img):
    H, S, V=cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eqV=cv2.equalizeHist(V)
    eqImage=cv2.cvtColor(cv2.merge([H, S, eqV]), cv2.COLOR_HSV2BGR)
    return eqImage

plt.figure(figsize=(18, 14))
plt.suptitle("Color histogram equalization with cv2.equalizeHist() in the V channel", fontsize=14, fontweight='bold')

image=cv2.imread('images/lenna.png')
histColor=HistColorImg(image)
imageEq=EqualizeHistColorHsv(image)
histImageEq=HistColorImg(imageEq)

M=np.ones(image.shape, dtype="uint8")*15
addedImage=cv2.add(image, M)
histColorAddedImage=HistColorImg(addedImage)
addedImageEq=EqualizeHistColorHsv(addedImage)
histAddedImageEq=HistColorImg(addedImageEq)
subtractedImage=cv2.subtract(image, M)
histColorSubtractedImage=HistColorImg(subtractedImage)
subtractedImageEq=EqualizeHistColorHsv(subtractedImage)
histSubtractedImageEq=HistColorImg(subtractedImageEq)

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