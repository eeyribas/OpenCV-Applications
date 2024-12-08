import cv2
import matplotlib.pyplot as plt

def ShowWithMatplotlib(colorImg, title, pos):
    imgRGB = colorImg[:, :, ::-1]
    ax = plt.subplot(2, 4, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

def SketchImage(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.medianBlur(imgGray, 5)
    edges = cv2.Laplacian(imgGray, cv2.CV_8U, ksize=5)
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    return thresholded

def cartonize_image(img, grayMode=False):
    thresholded = SketchImage(img)
    filtered = cv2.bilateralFilter(img, 10, 250, 250)
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)
    if grayMode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)
    return cartoonized

plt.figure(figsize=(14, 6))
plt.suptitle("Cartoonizing images", fontsize=14, fontweight='bold')

image = cv2.imread('images/cat.jpg')
customSketchImage = SketchImage(image)
customCartonizedImage = cartonize_image(image)
customCartonizedImageGray = cartonize_image(image, True)
sketchGray, sketchColor = cv2.pencilSketch(image, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
stylizatedImage = cv2.stylization(image, sigma_s=60, sigma_r=0.07)

ShowWithMatplotlib(image, "image", 1)
ShowWithMatplotlib(cv2.cvtColor(customSketchImage, cv2.COLOR_GRAY2BGR), "custom sketch", 2)
ShowWithMatplotlib(cv2.cvtColor(sketchGray, cv2.COLOR_GRAY2BGR), "sketch gray cv2.pencilSketch()", 3)
ShowWithMatplotlib(sketchColor, "sketch color cv2.pencilSketch()", 4)
ShowWithMatplotlib(stylizatedImage, "cartoonized cv2.stylization()", 5)
ShowWithMatplotlib(customCartonizedImage, "custom cartoonized", 6)
ShowWithMatplotlib(cv2.cvtColor(customCartonizedImageGray, cv2.COLOR_GRAY2BGR), "custom cartoonized gray", 7)
plt.show()