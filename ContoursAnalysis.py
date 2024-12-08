import numpy as np
import cv2
from matplotlib import pyplot as plt

def AspectRatio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    res = float(w) / h
    return res

def Roundness(contour, moments):
    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k

def EccentricityFromEllipse(contour):
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    a = ma / 2
    b = MA / 2
    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc

def EccentricityFromMoments(moments):
    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc

def GetOneContour():
    cnts = [np.array([[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
                             [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts

def ArrayToTuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def DrawContourPoints(img, cnts, color):
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            pp = ArrayToTuple(p)
            cv2.circle(img, pp, 10, color, -1)
    return img

def DrawContourOutline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contour analysis", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

canvas = np.zeros((640, 640, 3), dtype="uint8")
contours = GetOneContour()
print("'detected' contours: '{}' ".format(len(contours)))

imageContourPoints = canvas.copy()
imageContourOutline = canvas.copy()
imageContourPointsOutline = canvas.copy()

DrawContourPoints(imageContourPoints, contours, (255, 0, 255))
DrawContourOutline(imageContourOutline, contours, (0, 255, 255), -1)
DrawContourOutline(imageContourPointsOutline, contours, (255, 0, 0), 3)
DrawContourPoints(imageContourPointsOutline, contours, (0, 0, 255))
M = cv2.moments(contours[0])
print("moments calculated from the detected contour: {}".format(M))
print("Contour area: '{}'".format(cv2.contourArea(contours[0])))
print("Contour area: '{}'".format(M['m00']))

xCentroid = round(M['m10'] / M['m00'])
yCentroid = round(M['m01'] / M['m00'])
print("center X : '{}'".format(xCentroid))
print("center Y : '{}'".format(yCentroid))

cv2.circle(imageContourPoints, (xCentroid, yCentroid), 10, (255, 255, 255), -1)
k = Roundness(contours[0], M)
print("roundness: '{}'".format(k))
em = EccentricityFromMoments(M)
print("eccentricity: '{}'".format(em))
ee = EccentricityFromEllipse(contours[0])
print("eccentricity: '{}'".format(ee))
ar = AspectRatio(contours[0])
print("aspect ratio: '{}'".format(ar))

ShowImgWithMatplotlib(imageContourPoints, "centroid : (" + str(xCentroid) + "," + str(yCentroid) + ")", 1)
ShowImgWithMatplotlib(imageContourOutline, "size: " + str(M['m00']) + " & aspect ratio: " + str(ar), 2)
ShowImgWithMatplotlib(imageContourPointsOutline, "roundness: " + str(round(k, 3)) + " & eccentricity: " + str(round(ee, 3)), 3)

plt.show()