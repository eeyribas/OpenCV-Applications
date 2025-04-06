import numpy as np
import cv2
from matplotlib import pyplot as plt

def GetOneContour():
    cnts=[np.array([[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
                           [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts

def ArrayToTuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def DrawContourPoints(img, cnts, color):
    for cnt in cnts:
        squeeze=np.squeeze(cnt)
        for p in squeeze:
            p=ArrayToTuple(p)
            cv2.circle(img, p, 10, color, -1)
    return img

def DrawContourOutline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(1, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig=plt.figure(figsize=(12, 5))
plt.suptitle("Contours introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
canvas=np.zeros((640, 640, 3), dtype="uint8")
contours=GetOneContour()
print("contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}' ".format(len(contours)))

imageContourPoints=canvas.copy()
imageContourOutline=canvas.copy()
imageContourPointsOutline=canvas.copy()

DrawContourPoints(imageContourPoints, contours, (255, 0, 255))
DrawContourOutline(imageContourOutline, contours, (0, 255, 255), 3)
DrawContourOutline(imageContourPointsOutline, contours, (255, 0, 0), 3)
DrawContourPoints(imageContourPointsOutline, contours, (0, 0, 255))

ShowImgWithMatplotlib(imageContourPoints, "contour points", 1)
ShowImgWithMatplotlib(imageContourOutline, "contour outline", 2)
ShowImgWithMatplotlib(imageContourPointsOutline, "contour outline and points", 3)
plt.show()