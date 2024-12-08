import numpy as np
import cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax = plt.subplot(1, 4, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(18, 6))
plt.suptitle("Comparing histogram (OpenCV, numpy, matplotlib)", fontsize=14, fontweight='bold')
image = cv2.imread('images/lenna.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start = timer()
hist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])
end = timer()
execTimeCalcHist = (end - start) * 1000

start = timer()
histNp, binsNp = np.histogram(grayImage.ravel(), 256, [0, 256])
end = timer()
execTimeNpHist = (end - start) * 1000

start = timer()
(n, bins, patches) = plt.hist(grayImage.ravel(), 256, [0, 256])
end = timer()
execTimePltHist = (end - start) * 1000

ShowImgWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray", 1)
ShowHistWithMatplotlibGray(hist, "grayscale histogram (OpenCV)-" + str('% 6.2f ms' % execTimeCalcHist), 2, 'm')
ShowHistWithMatplotlibGray(histNp, "grayscale histogram (Numpy)-" + str('% 6.2f ms' % execTimeNpHist), 3, 'm')
ShowHistWithMatplotlibGray(n, "grayscale histogram (Matplotlib)-" + str('% 6.2f ms' % execTimePltHist), 4, 'm')

plt.show()