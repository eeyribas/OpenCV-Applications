import cv2

grayImg=cv2.imread('images/logo.png', cv2.IMREAD_GRAYSCALE)
dimensions=grayImg.shape
print(dimensions)

(h, w)=grayImg.shape
print("Dimensions of the image - Height: {}, Width: {}".format(h, w))

totalNumberOfPixels=grayImg.size
print("Total number of elements: {}".format(totalNumberOfPixels))
print("Total number of elements: {}".format(h*w))

imageDtype=grayImg.dtype
print("Image datatype: {}".format(imageDtype))

cv2.imshow("original image", grayImg)
cv2.waitKey(0)

i=grayImg[6, 40]
print("Pixel at (6,40) - Intensity: {}".format(i))

grayImg[6, 40]=0
i=grayImg[6, 40]
print("Pixel at (6,40) - Intensity: {}".format(i))

topLeftCorner=grayImg[0:50, 0:50]
cv2.imshow("top left corner original", topLeftCorner)
cv2.waitKey(0)

grayImg[20:70, 20:70]=topLeftCorner
cv2.imshow("modified image", grayImg)
cv2.waitKey(0)

grayImg[0:50, 0:50]=255
cv2.imshow("modified image", grayImg)
cv2.waitKey(0)