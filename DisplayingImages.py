import cv2
import numpy as np
import matplotlib.pyplot as plt

imgOpenCV = cv2.imread('images/logo.png')
b, g, r = cv2.split(imgOpenCV)
imgMatplotlib = cv2.merge([r, g, b])

plt.subplot(121)
plt.imshow(imgOpenCV)
plt.title('img OpenCV')

plt.subplot(122)
plt.imshow(imgMatplotlib)
plt.title('img matplotlib')
plt.show()

cv2.imshow('bgr image', imgOpenCV)
cv2.imshow('rgb image', imgMatplotlib)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgConcats = np.concatenate((imgOpenCV, imgMatplotlib), axis=1)
cv2.imshow('bgr image and rgb image', imgConcats)
cv2.waitKey(0)
cv2.destroyAllWindows()

B = imgOpenCV[:, :, 0]
G = imgOpenCV[:, :, 1]
R = imgOpenCV[:, :, 2]
imgRGB = imgOpenCV[:, :, ::-1]

cv2.imshow('img RGB (wrong color)', imgRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()