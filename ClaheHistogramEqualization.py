import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 5, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def EqualizeClaheColorHsv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eqV = cla.apply(V)
    eqImage = cv2.cvtColor(cv2.merge([H, S, eqV]), cv2.COLOR_HSV2BGR)
    return eqImage

def EqualizeClaheColorLab(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eqL = cla.apply(L)
    eqImage = cv2.cvtColor(cv2.merge([eqL, a, b]), cv2.COLOR_Lab2BGR)
    return eqImage

def EqualizeClaheColorYuv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eqY = cla.apply(Y)
    eqImage = cv2.cvtColor(cv2.merge([eqY, U, V]), cv2.COLOR_YUV2BGR)
    return eqImage

def EqualizeClaheColor(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eqChannels = []
    for ch in channels:
        eqChannels.append(cla.apply(ch))
    eqImage = cv2.merge(eqChannels)
    return eqImage

plt.figure(figsize=(18, 14))
plt.suptitle("Histogram equalization using CLAHE", fontsize=16, fontweight='bold')

image = cv2.imread('images/lenna.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0)
grayImageClahe = clahe.apply(grayImage)
clahe.setClipLimit(5.0)
grayImageClahe2 = clahe.apply(grayImage)
clahe.setClipLimit(10.0)
grayImageClahe3 = clahe.apply(grayImage)
clahe.setClipLimit(20.0)
grayImageClahe4 = clahe.apply(grayImage)

imageClaheColor = EqualizeClaheColor(image)
imageClaheColorLab = EqualizeClaheColorLab(image)
imageClaheColorHsv = EqualizeClaheColorHsv(image)
imageClaheColorYuv = EqualizeClaheColorYuv(image)

ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowImgWithMatplotlib(cv2.cvtColor(grayImageClahe, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=2.0", 2)
ShowImgWithMatplotlib(cv2.cvtColor(grayImageClahe2, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=5.0", 3)
ShowImgWithMatplotlib(cv2.cvtColor(grayImageClahe3, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=10.0", 4)
ShowImgWithMatplotlib(cv2.cvtColor(grayImageClahe4, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=20.0", 5)
ShowImgWithMatplotlib(image, "color", 6)
ShowImgWithMatplotlib(imageClaheColor, "clahe on each channel (BGR)", 7)
ShowImgWithMatplotlib(imageClaheColorLab, "clahe on L channel (LAB)", 8)
ShowImgWithMatplotlib(imageClaheColorHsv, "clahe on V channel (HSV)", 9)
ShowImgWithMatplotlib(imageClaheColorYuv, "clahe on Y channel (YUV)", 10)
plt.show()