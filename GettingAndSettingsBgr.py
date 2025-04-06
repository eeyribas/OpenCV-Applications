import cv2

img=cv2.imread('images/logo.png')
dimensions=img.shape
print(dimensions)

(h, w, c)=img.shape
print("Dimensions of the image - Height: {}, Width: {}, Channels: {}".format(h, w, c))

totalNumberOfPixels=img.size
print("Total number of elements: {}".format(totalNumberOfPixels))
print("Total number of elements: {}".format(h*w*c))

imageDtype=img.dtype
print("Image datatype: {}".format(imageDtype))

cv2.imshow("original image", img)
cv2.waitKey(0)
(b, g, r)=img[6, 40]
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

b=img[6, 40, 0]
g=img[6, 40, 1]
r=img[6, 40, 2]
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

img[6, 40]=(0, 0, 255)
(b, g, r)=img[6, 40]
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

topLeftCorner=img[0:50, 0:50]
cv2.imshow("top left corner original", topLeftCorner)
cv2.waitKey(0)

img[20:70, 20:70]=topLeftCorner
cv2.imshow("modified image", img)
cv2.waitKey(0)

img[0:50, 0:50]=(255, 0, 0)
cv2.imshow("modified image", img)
cv2.waitKey(0)