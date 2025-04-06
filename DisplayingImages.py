import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread('images/logo.png')
b, g, r=cv2.split(img1)
img2=cv2.merge([r, g, b])

plt.subplot(121)
plt.imshow(img1)
plt.title('OpenCV - Image')
plt.subplot(122)
plt.imshow(img2)
plt.title('Matplotlib - Image')
plt.show()

cv2.imshow('bgr image', img1)
cv2.imshow('rgb image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img3=np.concatenate((img1, img2), axis=1)
cv2.imshow('bgr image and rgb image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

B=img1[:, :, 0]
G=img1[:, :, 1]
R=img1[:, :, 2]
img4=img1[:, :, ::-1]
cv2.imshow('img RGB (wrong color)', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()