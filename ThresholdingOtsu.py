import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color, t=-1):
    ax=plt.subplot(2, 2, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

fig=plt.figure(figsize=(10, 10))
plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image=cv2.imread('images/leaf.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist=cv2.calcHist([grayImage], [0], None, [256], [0, 256])
ret1, th1=cv2.threshold(grayImage, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

ShowImgWithMatplotlib(image, "image", 1)
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray img", 2)
ShowHistWithMatplotlibGray(hist, "grayscale histogram", 3, 'm', ret1)
ShowImgWithMatplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Otsu's binarization", 4)
plt.show()