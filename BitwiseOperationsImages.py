import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(6, 5))
plt.suptitle("Bitwise AND/OR between two images", fontsize=14, fontweight='bold')
image=cv2.imread('images/lenna-250.png')

binaryImage=cv2.imread('images/opencv-binary-logo-250.png')
bitwiseAnd=cv2.bitwise_and(image, binaryImage)
bitwiseOr=cv2.bitwise_or(image, binaryImage)

ShowWithMatplotlib(image, "image", 1)
ShowWithMatplotlib(binaryImage, "binary logo", 2)
ShowWithMatplotlib(bitwiseAnd, "AND operation", 3)
ShowWithMatplotlib(bitwiseOr, "OR operation", 4)
plt.show()