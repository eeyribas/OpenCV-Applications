import cv2
from matplotlib import pyplot as plt

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(1, 3, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig=plt.figure(figsize=(12, 4))
plt.suptitle("Thresholding BGR images", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
image=cv2.imread('images/cat.jpg')
ShowImgWithMatplotlib(image, "image", 1)

ret1, thresh1=cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
(b, g, r)=cv2.split(image)
ret2, thresh2=cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
ret3, thresh3=cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
ret4, thresh4=cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
bgrThresh=cv2.merge((thresh2, thresh3, thresh4))

ShowImgWithMatplotlib(thresh1, "threshold (120) BGR image", 2)
ShowImgWithMatplotlib(bgrThresh, "threshold (120) each channel and merge", 3)
plt.show()