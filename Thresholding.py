import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(9, 9))
plt.suptitle("Thresholding example", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('images/sudoku.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "img", 1)

ret1, thresh1 = cv2.threshold(grayImage, 60, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(grayImage, 70, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(grayImage, 80, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(grayImage, 90, 255, cv2.THRESH_BINARY)
ret5, thresh5 = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
ret6, thresh6 = cv2.threshold(grayImage, 110, 255, cv2.THRESH_BINARY)
ret7, thresh7 = cv2.threshold(grayImage, 120, 255, cv2.THRESH_BINARY)
ret8, thresh8 = cv2.threshold(grayImage, 130, 255, cv2.THRESH_BINARY)

ShowImgWithMatplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 60", 2)
ShowImgWithMatplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 70", 3)
ShowImgWithMatplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 80", 4)
ShowImgWithMatplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 90", 5)
ShowImgWithMatplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 100", 6)
ShowImgWithMatplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 110", 7)
ShowImgWithMatplotlib(cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "threshold = 120", 8)
ShowImgWithMatplotlib(cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR), "threshold = 130", 9)

plt.show()