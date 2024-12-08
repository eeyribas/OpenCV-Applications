import numpy as np
import cv2
from matplotlib import pyplot as plt

def GetPositionToDraw(text, point, fontFace, fontScale, thickness):
    text_size = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

def DetectShape(contour):
    perimeter = cv2.arcLength(contour, True)
    contourApprox = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(contourApprox) == 3:
        detected_shape = 'triangle'
    elif len(contourApprox) == 4:
        x, y, width, height = cv2.boundingRect(contourApprox)
        aspect_ratio = float(width) / height
        if 0.90 < aspect_ratio < 1.10:
            detected_shape = "square"
        else:
            detected_shape = "rectangle"
    elif len(contourApprox) == 5:
        detected_shape = "pentagon"
    elif len(contourApprox) == 6:
        detected_shape = "hexagon"
    else:
        detected_shape = "circle"
    return detected_shape, contourApprox

def ArrayToTuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def DrawContourPoints(img, cnts, color):
    for cnt in cnts:
        print(cnt.shape)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)
        for p in squeeze:
            pp = ArrayToTuple(p)
            cv2.circle(img, pp, 10, color, -1)
    return img

def DrawContourOutline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 9))
plt.suptitle("Shape recognition based on cv2.approxPolyDP()", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("images/shapes.png")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("detected contours: '{}' ".format(len(contours)))

imageContours = image.copy()
imageRecognitionShapes = image.copy()
DrawContourOutline(imageContours, contours, (255, 255, 255), 4)

for contour in contours:
    M = cv2.moments(contour)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    shape, vertices = DetectShape(contour)
    DrawContourPoints(imageContours, [vertices], (255, 255, 255))
    (x, y) = GetPositionToDraw(shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(imageRecognitionShapes, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

ShowImgWithMatplotlib(image, "image", 1)
ShowImgWithMatplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
ShowImgWithMatplotlib(imageContours, "contours outline (after approximation)", 3)
ShowImgWithMatplotlib(imageRecognitionShapes, "contours recognition", 4)

plt.show()