import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 7, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

grayImg=cv2.imread('images/lenna.png', cv2.IMREAD_GRAYSCALE)
colormaps=["AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "HOT", "PINK", "PARULA"]
plt.figure(figsize=(12, 5))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')
ShowWithMatplotlib(cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR), "GRAY", 1)

for idx, val in enumerate(colormaps):
    ShowWithMatplotlib(cv2.applyColorMap(grayImg, idx), val, idx+2)
plt.show()