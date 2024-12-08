import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def UnsharpedFilter(img):
    smoothed = cv2.GaussianBlur(img, (9, 9), 10)
    return cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)

plt.figure(figsize=(12, 6))
plt.suptitle("Sharpening images", fontsize=14, fontweight='bold')
image = cv2.imread('images/cat-face.png')

kernelSharpen1 = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
kernelSharpen2 = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
kernelSharpen3 = np.array([[1, 1, 1],
                           [1, -7, 1],
                           [1, 1, 1]])
kernelSharpen4 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 2, 2, 2, -1],
                           [-1, 2, 8, 2, -1],
                           [-1, 2, 2, 2, -1],
                           [-1, -1, -1, -1, -1]]) / 8.0

sharpImage1 = cv2.filter2D(image, -1, kernelSharpen1)
sharpImage2 = cv2.filter2D(image, -1, kernelSharpen2)
sharpImage3 = cv2.filter2D(image, -1, kernelSharpen3)
sharpImage4 = cv2.filter2D(image, -1, kernelSharpen4)
sharpImage5 = UnsharpedFilter(image)

ShowWithMatplotlib(image, "original", 1)
ShowWithMatplotlib(sharpImage1, "sharp 1", 2)
ShowWithMatplotlib(sharpImage2, "sharp 2", 3)
ShowWithMatplotlib(sharpImage3, "sharp 3", 4)
ShowWithMatplotlib(sharpImage4, "sharp 4", 5)
ShowWithMatplotlib(sharpImage5, "sharp 5", 6)

plt.show()