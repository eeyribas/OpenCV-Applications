import numpy as np
import cv2
from matplotlib import pyplot as plt

def GetPositionToDraw(text, point, font_face, font_scale, thickness):
    textSize = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    textX = point[0] - textSize[0] / 2
    textY = point[1] + textSize[1] / 2
    return round(textX), round(textY)

def BuildCircleImage():
    img = np.zeros((500, 500, 3), dtype="uint8")
    cv2.circle(img, (250, 250), 200, (255, 255, 255), 1)
    return img

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(18, 6))
plt.suptitle("Matching contours (against a perfect circle) using cv2.matchShapes()", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("images/match-shapes.png")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageCircle = BuildCircleImage()
grayImageCircle = cv2.cvtColor(imageCircle, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY_INV)
ret, threshCircle = cv2.threshold(grayImageCircle, 70, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_circle, hierarchy_2 = cv2.findContours(threshCircle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
result1 = image.copy()
result2 = image.copy()
result3 = image.copy()

for contour in contours:
    M = cv2.moments(contour)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    ret1 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I1, 0.0)
    ret2 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I2, 0.0)
    ret3 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I3, 0.0)

    (x1, y1) = GetPositionToDraw(str(round(ret1, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x2, y2) = GetPositionToDraw(str(round(ret2, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x3, y3) = GetPositionToDraw(str(round(ret3, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    cv2.putText(result1, str(round(ret1, 3)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.putText(result2, str(round(ret2, 3)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(result3, str(round(ret3, 3)), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

ShowImgWithMatplotlib(result1, "matching scores (method = CONTOURS_MATCH_I1)", 1)
ShowImgWithMatplotlib(result2, "matching scores (method = CONTOURS_MATCH_I2)", 2)
ShowImgWithMatplotlib(result3, "matching scores (method = CONTOURS_MATCH_I3)", 3)

plt.show()