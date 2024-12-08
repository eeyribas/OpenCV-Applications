import numpy as np
import cv2
from matplotlib import pyplot as plt

def ArrayToTuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def DrawContourPoints(img, cnts, color):
    for cnt in cnts:
        print(cnt.shape)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)
        for p in squeeze:
            pp = ArrayToTuple(p)
            cv2.circle(img, pp, 3, color, -1)
    return img

def BuildSampleImage():
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)
    return img

def BuildSampleImage2():
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.rectangle(img, (150, 150), (250, 250), (70, 70, 70), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)
    cv2.circle(img, (400, 400), 50, (70, 70, 70), -1)
    return img

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 8))
plt.suptitle("Contours approximation method", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = BuildSampleImage2()
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY)

imageApproxNone = image.copy()
imageApproxSimple = image.copy()
imageApproxTc89L1 = image.copy()
imageApproxTc89Kcos = image.copy()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
contours4, hierarchy4 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

DrawContourPoints(imageApproxNone, contours, (255, 255, 255))
DrawContourPoints(imageApproxSimple, contours2, (255, 255, 255))
DrawContourPoints(imageApproxTc89L1, contours3, (255, 255, 255))
DrawContourPoints(imageApproxTc89Kcos, contours4, (255, 255, 255))

ShowImgWithMatplotlib(image, "image", 1)
ShowImgWithMatplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
ShowImgWithMatplotlib(imageApproxNone, "contours (APPROX_NONE)", 3)
ShowImgWithMatplotlib(imageApproxSimple, "contours (CHAIN_APPROX_SIMPLE)", 4)
ShowImgWithMatplotlib(imageApproxTc89L1, "contours (APPROX_TC89_L1)", 5)
ShowImgWithMatplotlib(imageApproxTc89Kcos, "contours (APPROX_TC89_KCOS)", 6)

plt.show()