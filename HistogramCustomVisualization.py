import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibRGB(hist, title, pos, color):
    ax=plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax=plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

def HistColorImg(img):
    histr=[]
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr

def PlotHist(histItems, color):
    offsetDown=10
    offsetUp=10
    xValues=np.arange(256).reshape(256, 1)
    canvas=np.ones((300, 256, 3), dtype="uint8")*255
    for histItem, col in zip(histItems, color):
        cv2.normalize(histItem, histItem, 0+offsetDown, 300-offsetUp, cv2.NORM_MINMAX)
        around=np.around(histItem)
        hist=np.int32(around)
        pts=np.column_stack((xValues, hist))
        cv2.polylines(canvas, [pts], False, col, 2)
        cv2.rectangle(canvas, (0, 0), (255, 298), (0, 0, 0), 1)
    res=np.flipud(canvas)
    return res

plt.figure(figsize=(16, 10))
plt.suptitle("Custom visualization of histograms", fontsize=14, fontweight='bold')

image=cv2.imread('images/lenna-mod.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([grayImage], [0], None, [256], [0, 256])
histColor=HistColorImg(image)
grayPlot=PlotHist([hist], [(255, 0, 255)])
colorPlot=PlotHist(histColor, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowImgWithMatplotlib(image, "image", 4)
ShowHistWithMatplotlibGray(hist, "grayscale histogram (matplotlib)", 2, 'm')
ShowHistWithMatplotlibRGB(histColor, "color histogram (matplotlib)", 3, ['b', 'g', 'r'])
ShowImgWithMatplotlib(grayPlot, "grayscale histogram (custom)", 5)
ShowImgWithMatplotlib(colorPlot, "color histogram (custom)", 6)
plt.show()