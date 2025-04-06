import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(1, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(10, 4))
plt.suptitle("Sobel operator and cv2.addWeighted() to show the output", fontsize=14, fontweight='bold')

image=cv2.imread('images/lenna.png')
imageFiltered=cv2.GaussianBlur(image, (3, 3), 0)

grayImage=cv2.cvtColor(imageFiltered, cv2.COLOR_BGR2GRAY)
gradientX=cv2.Sobel(grayImage, cv2.CV_16S, 1, 0, 3)
gradientY=cv2.Sobel(grayImage, cv2.CV_16S, 0, 1, 3)
absGradientX=cv2.convertScaleAbs(gradientX)
absGradientY=cv2.convertScaleAbs(gradientY)
sobelImage=cv2.addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0)

ShowWithMatplotlib(image, "Image", 1)
ShowWithMatplotlib(cv2.cvtColor(absGradientX, cv2.COLOR_GRAY2BGR), "Gradient x", 2)
ShowWithMatplotlib(cv2.cvtColor(absGradientY, cv2.COLOR_GRAY2BGR), "Gradient y", 3)
ShowWithMatplotlib(cv2.cvtColor(sobelImage, cv2.COLOR_GRAY2BGR), "Sobel output", 4)

plt.show()