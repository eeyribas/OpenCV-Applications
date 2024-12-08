import cv2

image = cv2.imread("images/logo.png")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("OpenCV logo", image)
cv2.imshow("OpenCV logo gray format", grayImage)

cv2.waitKey(0)
cv2.destroyAllWindows()