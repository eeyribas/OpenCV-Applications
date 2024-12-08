import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 4))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')
grayImg = cv2.imread('images/lenna.png', cv2.IMREAD_GRAYSCALE)
imgCOLORMAPHSV = cv2.applyColorMap(grayImg, cv2.COLORMAP_HSV)
ShowWithMatplotlib(cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR), "gray image", 1)
ShowWithMatplotlib(imgCOLORMAPHSV, "HSV", 2)

plt.show()