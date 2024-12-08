import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Hu moment invariants properties", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image1 = cv2.imread("images/shape-features.png")
image2 = cv2.imread("images/shape-features-rotation.png")
image3 = cv2.imread("images/shape-features-reflection.png")
grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
grayImage3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(grayImage1, 70, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(grayImage2, 70, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(grayImage3, 70, 255, cv2.THRESH_BINARY)
HuM1 = cv2.HuMoments(cv2.moments(thresh1, True)).flatten()
HuM2 = cv2.HuMoments(cv2.moments(thresh2, True)).flatten()
HuM3 = cv2.HuMoments(cv2.moments(thresh3, True)).flatten()
print("Hu moments (original): '{}'".format(HuM1))
print("Hu moments (rotation): '{}'".format(HuM2))
print("Hu moments (reflection): '{}'".format(HuM3))

ShowImgWithMatplotlib(image1, "original", 1)
ShowImgWithMatplotlib(image2, "rotation", 2)
ShowImgWithMatplotlib(image3, "reflection", 3)

plt.show()