import numpy as np
import cv2
from matplotlib import pyplot as plt

def DrawContourOutline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

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
    ax = plt.subplot(1, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contours introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = BuildSampleImage2()
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print("detected contours (RETR_EXTERNAL): '{}' ".format(len(contours)))
print("detected contours (RETR_LIST): '{}' ".format(len(contours2)))

imageContours = image.copy()
imageContours2 = image.copy()
DrawContourOutline(imageContours, contours, (0, 0, 255), 5)
DrawContourOutline(imageContours2, contours2, (255, 0, 0), 5)

ShowImgWithMatplotlib(image, "image", 1)
ShowImgWithMatplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
ShowImgWithMatplotlib(imageContours, "contours (RETR EXTERNAL)", 3)
ShowImgWithMatplotlib(imageContours2, "contours (RETR LIST)", 4)

plt.show()