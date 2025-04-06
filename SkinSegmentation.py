import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

imageNames=['skin-test-img-1.jpg', 'skin-test-img-2.jpg', 'skin-test-img-3.jpg',
            'skin-test-img-4.jpg', 'skin-test-img-5.jpg', 'skin-test-img-6.jpg']
path='images'

def LoadAllTestImages():
    skinImages=[]
    for indexImage, nameImage in enumerate(imageNames):
        imagePath=os.path.join(path, nameImage)
        skinImages.append(cv2.imread(imagePath))
    return skinImages

def ShowImages(arrayImg, title, pos):
    for indexImage, image in enumerate(arrayImg):
        ShowWithMatplotlib(image, title+"_"+str(indexImage+1), pos+indexImage)

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(5, 6, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

lowerHsv1=np.array([0, 48, 80], dtype="uint8")
upperHsv1=np.array([20, 255, 255], dtype="uint8")

def SkinDetectorHsv(bgrImage):
    hsvImage=cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)
    skinRegion=cv2.inRange(hsvImage, lowerHsv1, upperHsv1)
    return skinRegion

lowerHsv2= np.array([0, 50, 0], dtype="uint8")
upperHsv2= np.array([120, 150, 255], dtype="uint8")

def SkinDetectorHsv2(bgrImage):
    hsvImage=cv2.cvtColor(bgrImage, cv2.COLOR_BGR2HSV)
    skinRegion=cv2.inRange(hsvImage, lowerHsv2, upperHsv2)
    return skinRegion

lowerYcrcb=np.array([0, 133, 77], dtype="uint8")
upperYcrcb=np.array([255, 173, 127], dtype="uint8")

def SkinDetectorYcrcb(bgrImage):
    ycrcbImage=cv2.cvtColor(bgrImage, cv2.COLOR_BGR2YCR_CB)
    skinRegion=cv2.inRange(ycrcbImage, lowerYcrcb, upperYcrcb)
    return skinRegion

def BgrSkin(b, g, r):
    e1=bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
            abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2=bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2

def SkinDetectorBgr(bgrImage):
    h=bgrImage.shape[0]
    w=bgrImage.shape[1]
    res=np.zeros((h, w, 1), dtype="uint8")
    for y in range(0, h):
        for x in range(0, w):
            (b, g, r)=bgrImage[y, x]
            if BgrSkin(b, g, r):
                res[y, x]=255
    return res

SkinDetectors={
    'ycrcb':SkinDetectorYcrcb,
    'hsv':SkinDetectorHsv,
    'hsv_2':SkinDetectorHsv2,
    'bgr':SkinDetectorBgr
}

def ApplySkinDetector(arrayImg, skinDetector):
    skinDetectorResult=[]
    for indexImage, image in enumerate(arrayImg):
        detectedSkin=SkinDetectors[skinDetector](image)
        bgr=cv2.cvtColor(detectedSkin, cv2.COLOR_GRAY2BGR)
        skinDetectorResult.append(bgr)
    return skinDetectorResult

plt.figure(figsize=(15, 8))
plt.suptitle("Skin segmentation using different color spaces", fontsize=14, fontweight='bold')

for i, (k, v) in enumerate(SkinDetectors.items()):
    print("index: '{}', key: '{}', value: '{}'".format(i, k, v))

testImages=LoadAllTestImages()
ShowImages(testImages, "test img", 1)

ShowImages(ApplySkinDetector(testImages, 'ycrcb'), "ycrcb", 7)
ShowImages(ApplySkinDetector(testImages, 'hsv'), "hsv", 13)
ShowImages(ApplySkinDetector(testImages, 'hsv_2'), "hsv_2", 19)
ShowImages(ApplySkinDetector(testImages, 'bgr'), "bgr", 25)
plt.show()