import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(3, 6, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

image = cv2.imread('images/color-spaces.png')
plt.figure(figsize=(12, 5))
plt.suptitle("Color spaces in OpenCV", fontsize=14, fontweight='bold')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(bgrB, bgrG, bgrR) = cv2.split(image)
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
(hsvH, hsvS, hsvV) = cv2.split(hsvImage)
hlsImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
(hlsH, hlsL, hlsS) = cv2.split(hlsImage)
ycrcbImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
(ycrcbY, ycrcbCr, ycrcbCb) = cv2.split(ycrcbImage)
labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
(labL, labA, labB) = cv2.split(labImage)

ShowWithMatplotlib(image, "BGR - image", 1)
ShowWithMatplotlib(cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR), "gray image", 1 + 6)
ShowWithMatplotlib(cv2.cvtColor(bgrB, cv2.COLOR_GRAY2BGR), "BGR - B comp", 2)
ShowWithMatplotlib(cv2.cvtColor(bgrG, cv2.COLOR_GRAY2BGR), "BGR - G comp", 2 + 6)
ShowWithMatplotlib(cv2.cvtColor(bgrR, cv2.COLOR_GRAY2BGR), "BGR - R comp", 2 + 6 * 2)

ShowWithMatplotlib(cv2.cvtColor(hsvH, cv2.COLOR_GRAY2BGR), "HSV - H comp", 3)
ShowWithMatplotlib(cv2.cvtColor(hsvS, cv2.COLOR_GRAY2BGR), "HSV - S comp", 3 + 6)
ShowWithMatplotlib(cv2.cvtColor(hsvV, cv2.COLOR_GRAY2BGR), "HSV - V comp", 3 + 6 * 2)

ShowWithMatplotlib(cv2.cvtColor(hlsH, cv2.COLOR_GRAY2BGR), "HLS - H comp", 4)
ShowWithMatplotlib(cv2.cvtColor(hlsL, cv2.COLOR_GRAY2BGR), "HLS - L comp", 4 + 6)
ShowWithMatplotlib(cv2.cvtColor(hlsS, cv2.COLOR_GRAY2BGR), "HLS - S comp", 4 + 6 * 2)

ShowWithMatplotlib(cv2.cvtColor(ycrcbY, cv2.COLOR_GRAY2BGR), "YCrCb - Y comp", 5)
ShowWithMatplotlib(cv2.cvtColor(ycrcbCr, cv2.COLOR_GRAY2BGR), "YCrCb - Cr comp", 5 + 6)
ShowWithMatplotlib(cv2.cvtColor(ycrcbCb, cv2.COLOR_GRAY2BGR), "YCrCb - Cb comp", 5 + 6 * 2)

ShowWithMatplotlib(cv2.cvtColor(labL, cv2.COLOR_GRAY2BGR), "L*a*b - L comp", 6)
ShowWithMatplotlib(cv2.cvtColor(labA, cv2.COLOR_GRAY2BGR), "L*a*b - a comp", 6 + 6)
ShowWithMatplotlib(cv2.cvtColor(labB, cv2.COLOR_GRAY2BGR), "L*a*b - b comp", 6 + 6 * 2)

plt.show()