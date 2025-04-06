import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax=plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(14, 8))
plt.suptitle("Grayscale histogram equalization with cv2.calcHist() and CLAHE", fontsize=16, fontweight='bold')

image=cv2.imread('images/lenna.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([grayImage], [0], None, [256], [0, 256])
grayImageEq=cv2.equalizeHist(grayImage)
histEq=cv2.calcHist([grayImageEq], [0], None, [256], [0, 256])

clahe=cv2.createCLAHE(clipLimit=4.0)
grayImageClahe=clahe.apply(grayImage)
histClahe=cv2.calcHist([grayImageClahe], [0], None, [256], [0, 256])

ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 4, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(grayImageEq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2)
ShowHistWithMatplotlibGray(histEq, "grayscale equalized histogram", 5, 'm')
ShowImgWithMatplotlib(cv2.cvtColor(grayImageClahe, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3)
ShowHistWithMatplotlibGray(histClahe, "grayscale clahe histogram", 6, 'm')
plt.show()