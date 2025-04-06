import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(3, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color, otsu=-1):
    ax=plt.subplot(3, 2, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.axvline(x=otsu, color='m', linestyle='--')
    plt.plot(hist, color=color)

fig=plt.figure(figsize=(11, 10))
plt.suptitle("Triangle binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image=cv2.imread('images/leaf-noise.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([grayImage], [0], None, [256], [0, 256])
ret1, th1=cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
grayImageBlurred=cv2.GaussianBlur(grayImage, (25, 25), 0)
hist2=cv2.calcHist([grayImageBlurred], [0], None, [256], [0, 256])
ret2, th2=cv2.threshold(grayImageBlurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

ShowImgWithMatplotlib(image, "image with noise", 1)
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray img with noise", 2)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 3, 'm', ret1)
ShowImgWithMatplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Triangle binarization (before applying a Gaussian filter)", 4)
ShowHistWithMatplotlibGray(hist2, "grayscale histogram", 5, 'm', ret2)
ShowImgWithMatplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR), "Triangle binarization (after applying a Gaussian filter)", 6)
plt.show()