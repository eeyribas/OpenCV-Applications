import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Smoothing techniques", fontsize=14, fontweight='bold')
image = cv2.imread('images/cat-face.png')
kernelAveraging1010 = np.ones((10, 10), np.float32) / 100

kernelAveraging55 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
print("kernel: {}".format(kernelAveraging55))

smoothImageF2D55 = cv2.filter2D(image, -1, kernelAveraging55)
smoothImageF2D1010 = cv2.filter2D(image, -1, kernelAveraging1010)
smoothImageB = cv2.blur(image, (10, 10))
smoothImageBfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)
smoothImageGb = cv2.GaussianBlur(image, (9, 9), 0)
smoothImageMb = cv2.medianBlur(image, 9)
smoothImageBf = cv2.bilateralFilter(image, 5, 10, 10)
smoothImageBf2 = cv2.bilateralFilter(image, 9, 200, 200)

ShowWithMatplotlib(image, "original", 1)
ShowWithMatplotlib(smoothImageF2D55, "cv2.filter2D() (5,5) kernel", 2)
ShowWithMatplotlib(smoothImageF2D1010, "cv2.filter2D() (10,10) kernel", 3)
ShowWithMatplotlib(smoothImageB, "cv2.blur()", 4)
ShowWithMatplotlib(smoothImageBfi, "cv2.boxFilter()", 5)
ShowWithMatplotlib(smoothImageGb, "cv2.GaussianBlur()", 6)
ShowWithMatplotlib(smoothImageMb, "cv2.medianBlur()", 7)
ShowWithMatplotlib(smoothImageBf, "cv2.bilateralFilter() - small values", 8)
ShowWithMatplotlib(smoothImageBf2, "cv2.bilateralFilter() - big values", 9)
plt.show()