import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(3, 6, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

image=cv2.imread('images/color-spaces.png')
plt.figure(figsize=(13, 5))
plt.suptitle("Splitting and merging channels in OpenCV", fontsize=14, fontweight='bold')
ShowWithMatplotlib(image, "BGR - image", 1)

(b, g, r)=cv2.split(image)
ShowWithMatplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR - (B)", 2)
ShowWithMatplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR - (G)", 2+6)
ShowWithMatplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR - (R)", 2+6*2)

imageCopy=cv2.merge((b, g, r))
ShowWithMatplotlib(imageCopy, "BGR - image (copy)", 1+6)
bCopy=image[:, :, 0]

imageWithoutBlue=image.copy()
imageWithoutBlue[:, :, 0]=0
imageWithoutGreen=image.copy()
imageWithoutGreen[:, :, 1]=0
imageWithoutRed=image.copy()
imageWithoutRed[:, :, 2]=0

ShowWithMatplotlib(imageWithoutBlue, "BGR without B", 3)
ShowWithMatplotlib(imageWithoutGreen, "BGR without G", 3+6)
ShowWithMatplotlib(imageWithoutRed, "BGR without R", 3+6*2)

(b, g, r)=cv2.split(imageWithoutBlue)
ShowWithMatplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without B (B)", 4)
ShowWithMatplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without B (G)", 4+6)
ShowWithMatplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without B (R)", 4+6*2)
(b, g, r)=cv2.split(imageWithoutGreen)
ShowWithMatplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without G (B)", 5)
ShowWithMatplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without G (G)", 5+6)
ShowWithMatplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without G (R)", 5+6*2)
(b, g, r)=cv2.split(imageWithoutRed)
ShowWithMatplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 6)
ShowWithMatplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 6+6)
ShowWithMatplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 6+6*2)
plt.show()