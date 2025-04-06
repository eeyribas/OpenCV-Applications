import cv2
import matplotlib.pyplot as plt
import os

imageNames=['morpho-test-img-1.png', 'morpho-test-img-2.png', 'morpho-test-img-3.png']
path='images'
kernelSize33=(3, 3)
kernelSize55=(5, 5)

def LoadAllTestImages():
    testMorphImages=[]
    for indexImage, nameImage in enumerate(imageNames):
        imagePath=os.path.join(path, nameImage)
        testMorphImages.append(cv2.imread(imagePath))
    return testMorphImages

def ShowImages(arrayImg, title, pos):
    for indexImage, image in enumerate(arrayImg):
        ShowWithMatplotlib(image, title+"_"+str(indexImage+1), pos+indexImage*(len(MorphologicalOperations)+1))

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(len(imageNames), len(MorphologicalOperations)+1, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def BuildKernel(kernelType, kernelSize):
    if kernelType==cv2.MORPH_ELLIPSE:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
    elif kernelType==cv2.MORPH_CROSS:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernelSize)
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

def Erode(image, kernelType, kernelSize):
    kernel=BuildKernel(kernelType, kernelSize)
    erosion=cv2.erode(image, kernel, iterations=1)
    return erosion

def Dilate(image, kernelType, kernelSize):
    kernel=BuildKernel(kernelType, kernelSize)
    dilation=cv2.dilate(image, kernel, iterations=1)
    return dilation

def Closing(image, kernelType, kernelSize):
    kernel=BuildKernel(kernelType, kernelSize)
    clos=cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return clos

def Opening(image, kernelType, kernelSize):
    kernel=BuildKernel(kernelType, kernelSize)
    ope=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return ope

def MorphologicalGradient(image, kernelType, kernelSize):
    kernel=BuildKernel(kernelType, kernelSize)
    morphGradient=cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return morphGradient

def ClosingAndOpening(image, kernelType, kernelSize):
    closingImg=Closing(image, kernelType, kernelSize)
    openingImg=Opening(closingImg, kernelType, kernelSize)
    return openingImg

def OpeningAndClosing(image, kernelType, kernelSize):
    openingImg=Opening(image, kernelType, kernelSize)
    closingImg=Closing(openingImg, kernelType, kernelSize)
    return closingImg

MorphologicalOperations={
    'erode':Erode,
    'dilate':Dilate,
    'closing':Closing,
    'opening':Opening,
    'gradient':MorphologicalGradient,
    'closing|opening':ClosingAndOpening,
    'opening|closing':OpeningAndClosing
}

def ApplyMorphologicalOperation(arrayImg, morphologicalOperation, kernelType, kernelSize):
    morphologicalOperationResult=[]
    for indexImage, image in enumerate(arrayImg):
        result=MorphologicalOperations[morphologicalOperation](image, kernelType, kernelSize)
        morphologicalOperationResult.append(result)
    return morphologicalOperationResult

for i, (k, v) in enumerate(MorphologicalOperations.items()):
    print("index: '{}', key: '{}', value: '{}'".format(i, k, v))

testImages=LoadAllTestImages()
plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(3,3)'", fontsize=14, fontweight='bold')
ShowImages(testImages, "test img", 1)

for i, (k, v) in enumerate(MorphologicalOperations.items()):
    ShowImages(ApplyMorphologicalOperation(testImages, k, cv2.MORPH_RECT, kernelSize33), k, i+2)
plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_RECT', kernel_size='(5,5)'", fontsize=14, fontweight='bold')
ShowImages(testImages, "test img", 1)
for i, (k, v) in enumerate(MorphologicalOperations.items()):
    ShowImages(ApplyMorphologicalOperation(testImages, k, cv2.MORPH_RECT, kernelSize55), k, i+2)
plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(3,3)'", fontsize=14, fontweight='bold')
ShowImages(testImages, "test img", 1)
for i, (k, v) in enumerate(MorphologicalOperations.items()):
    ShowImages(ApplyMorphologicalOperation(testImages, k, cv2.MORPH_CROSS, kernelSize33), k, i+2)
plt.show()

plt.figure(figsize=(16, 8))
plt.suptitle("Morpho operations - kernel_type='cv2.MORPH_CROSS', kernel_size='(5,5)'", fontsize=14, fontweight='bold')
ShowImages(testImages, "test img", 1)

for i, (k, v) in enumerate(MorphologicalOperations.items()):
    ShowImages(ApplyMorphologicalOperation(testImages, k, cv2.MORPH_CROSS, kernelSize55), k, i+2)
plt.show()