import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax=plt.subplot(2, 2, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(10, 6))
plt.suptitle("Grayscale masked histogram", fontsize=14, fontweight='bold')
image=cv2.imread('images/lenna-mod.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([grayImage], [0], None, [256], [0, 256])
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 2, 'm')

mask=np.zeros(grayImage.shape[:2], np.uint8)
mask[30:190, 30:190]=255
histMask=cv2.calcHist([grayImage], [0], mask, [256], [0, 256])
maskedImg=cv2.bitwise_and(grayImage, grayImage, mask=mask)
ShowImgWithMatplotlib(cv2.cvtColor(maskedImg, cv2.COLOR_GRAY2BGR), "masked gray image", 3)
ShowHistWithMatplotlibGray(histMask, "grayscale masked histogram", 4, 'm')
plt.show()