import cv2
from matplotlib import pyplot as plt
import os

imageNames=['gray-image.png', 'gray-blurred.png', 'gray-added-image.png', 'gray-subtracted-image.png']
path='images'

def LoadAllTestImages():
    images=[]
    for indexImage, nameImage in enumerate(imageNames):
        imagePath=os.path.join(path, nameImage)
        images.append(cv2.imread(imagePath, 0))
    return images

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(4, 5, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def ShowHistWithMatplotlibGray(hist, title, pos, color):
    ax=plt.subplot(2, 5, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(18, 9))
plt.suptitle("Grayscale histogram comparison", fontsize=14, fontweight='bold')
testImages=LoadAllTestImages()
hists=[]

for img in testImages:
    hist=cv2.calcHist([img], [0], None, [256], [0, 256])
    hist=cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    hists.append(hist)

grayGray=cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CORREL)
grayGrayBlurred=cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
grayAddedGray=cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CORREL)
graySubGray=cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CORREL)

ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "query img", 1)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "img 1 "+str('CORREL % 6.5f'%grayGray), 2)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[1], cv2.COLOR_GRAY2BGR), "img 2 "+str('CORREL % 6.5f'%grayGrayBlurred), 3)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[2], cv2.COLOR_GRAY2BGR), "img 3 "+str('CORREL % 6.5f'%grayAddedGray), 4)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[3], cv2.COLOR_GRAY2BGR), "img 4 "+str('CORREL % 6.5f'%graySubGray), 5)

grayGray=cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CHISQR)
grayGrayBlurred=cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CHISQR)
grayAddedGray=cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CHISQR)
graySubGray=cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CHISQR)

ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "query img", 6)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "img 1 "+str('CHISQR % 6.5f'%grayGray), 7)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[1], cv2.COLOR_GRAY2BGR), "img 2 "+str('CHISQR % 6.5f'%grayGrayBlurred), 8)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[2], cv2.COLOR_GRAY2BGR), "img 3 "+str('CHISQR % 6.5f'%grayAddedGray), 9)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[3], cv2.COLOR_GRAY2BGR), "img 4 "+str('CHISQR % 6.5f'%graySubGray), 10)

grayGray=cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_INTERSECT)
grayGrayBlurred=cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_INTERSECT)
grayAddedGray=cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_INTERSECT)
graySubGray=cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_INTERSECT)

ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "query img", 11)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "img 1 "+str('INTERSECT % 6.5f'%grayGray), 12)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[1], cv2.COLOR_GRAY2BGR), "img 2 "+str('INTERSECT % 6.5f'%grayGrayBlurred), 13)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[2], cv2.COLOR_GRAY2BGR), "img 3 "+str('INTERSECT % 6.5f'%grayAddedGray), 14)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[3], cv2.COLOR_GRAY2BGR), "img 4 "+str('INTERSECT % 6.5f'%graySubGray), 15)

grayGray=cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_BHATTACHARYYA)
grayGrayBlurred=cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_BHATTACHARYYA)
grayAddedGray=cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_BHATTACHARYYA)
graySubGray=cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_BHATTACHARYYA)

ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "query img", 16)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[0], cv2.COLOR_GRAY2BGR), "img 1 "+str('BHATTACHARYYA % 6.5f'%grayGray), 17)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[1], cv2.COLOR_GRAY2BGR), "img 2 "+str('BHATTACHARYYA % 6.5f'%grayGrayBlurred), 18)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[2], cv2.COLOR_GRAY2BGR), "img 3 "+str('BHATTACHARYYA % 6.5f'%grayAddedGray), 19)
ShowImgWithMatplotlib(cv2.cvtColor(testImages[3], cv2.COLOR_GRAY2BGR), "img 4 "+str('BHATTACHARYYA % 6.5f'%graySubGray), 20)
plt.show()