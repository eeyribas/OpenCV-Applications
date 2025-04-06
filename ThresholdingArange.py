import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(3, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig=plt.figure(figsize=(9, 9))
plt.suptitle("Thresholding using np.arange() to create the different threshold values", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image=cv2.imread('images/sudoku.png')
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "img", 1)

thresholdValues=np.arange(start=60, stop=140, step=10)
thresholdedImages=[]
for threshold in thresholdValues:
    ret, thresh=cv2.threshold(grayImage, threshold, 255, cv2.THRESH_BINARY)
    thresholdedImages.append(thresh)

for index, (thresholdedImage, thresholdValue) in enumerate(zip(thresholdedImages, thresholdValues)):
    ShowImgWithMatplotlib(cv2.cvtColor(thresholdedImage, cv2.COLOR_GRAY2BGR), "threshold = "+str(thresholdValue), index+2)
plt.show()