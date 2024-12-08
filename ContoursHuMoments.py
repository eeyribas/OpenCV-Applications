import cv2
from matplotlib import pyplot as plt

def Centroid(moments):
    xCentroid = round(moments['m10'] / moments['m00'])
    yCentroid = round(moments['m01'] / moments['m00'])
    return xCentroid, yCentroid

def DrawContourOutline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Hu moments", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("images/shape-features.png")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY)
M = cv2.moments(thresh, True)
print("moments: '{}'".format(M))

x, y = Centroid(M)
HuM = cv2.HuMoments(M)
print("Hu moments: '{}'".format(HuM))

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
M2 = cv2.moments(contours[0])
print("moments: '{}'".format(M2))

x2, y2 = Centroid(M2)
HuM2 = cv2.HuMoments(M2)
print("Hu moments: '{}'".format(HuM2))

DrawContourOutline(image, contours, (255, 0, 0), 10)
cv2.circle(image, (x, y), 25, (255, 0, 0), -1)
cv2.circle(image, (x2, y2), 25, (0, 255, 0), -1)
print("('x','y'): ('{}','{}')".format(x, y))
print("('x2','y2'): ('{}','{}')".format(x2, y2))

ShowImgWithMatplotlib(image, "detected contour and centroid", 1)
plt.show()