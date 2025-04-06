import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(3, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Comparing different kernels using cv2.filter2D()", fontsize=14, fontweight='bold')
image=cv2.imread('images/cat-face.png')

kernelIdentity=np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
kernelEdgeDetection1=np.array([[1, 0, -1],
                               [0, 0, 0],
                               [-1, 0, 1]])
kernelEdgeDetection2=np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
kernelEdgeDetection3=np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])
kernelSharpen=np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
kernelUnsharpMasking=-1/256*np.array([[1, 4, 6, 4, 1],
                                      [4, 16, 24, 16, 4],
                                      [6, 24, -476, 24, 6],
                                      [4, 16, 24, 16, 4],
                                      [1, 4, 6, 4, 1]])
kernelBlur=1/9*np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
gaussianBlur=1/16*np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])
kernelEmboss=np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
sobelXKernel=np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
sobelYKernel=np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
outlineKernel=np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

originalImage=cv2.filter2D(image, -1, kernelIdentity)
edgeImage1=cv2.filter2D(image, -1, kernelEdgeDetection1)
edgeImage2=cv2.filter2D(image, -1, kernelEdgeDetection2)
edgeImage3=cv2.filter2D(image, -1, kernelEdgeDetection3)
sharpenImage=cv2.filter2D(image, -1, kernelSharpen)
unsharpMaskingImage=cv2.filter2D(image, -1, kernelUnsharpMasking)
blurImage=cv2.filter2D(image, -1, kernelBlur)
gaussianBlurImage=cv2.filter2D(image, -1, gaussianBlur)
embossImage=cv2.filter2D(image, -1, kernelEmboss)
sobelXImage=cv2.filter2D(image, -1, sobelXKernel)
sobelYImage=cv2.filter2D(image, -1, sobelYKernel)
outlineImage=cv2.filter2D(image, -1, outlineKernel)

ShowWithMatplotlib(originalImage, "identity kernel", 1)
ShowWithMatplotlib(edgeImage1, "edge detection 1", 2)
ShowWithMatplotlib(edgeImage2, "edge detection 2", 3)
ShowWithMatplotlib(edgeImage3, "edge detection 3", 4)
ShowWithMatplotlib(sharpenImage, "sharpen", 5)
ShowWithMatplotlib(unsharpMaskingImage, "unsharp masking", 6)
ShowWithMatplotlib(blurImage, "blur image", 7)
ShowWithMatplotlib(gaussianBlurImage, "gaussian blur image", 8)
ShowWithMatplotlib(embossImage, "emboss image", 9)
ShowWithMatplotlib(sobelXImage, "sobel x image", 10)
ShowWithMatplotlib(sobelYImage, "sobel y image", 11)
ShowWithMatplotlib(outlineImage, "outline image", 12)
plt.show()