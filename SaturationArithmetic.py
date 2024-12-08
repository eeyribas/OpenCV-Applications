import numpy as np
import cv2

x = np.uint8([250])
y = np.uint8([50])

resultOpencv = cv2.add(x, y)
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, resultOpencv))
resultNumpy = x + y
print("x:'{}' + y:'{}' = '{}'".format(x, y, resultNumpy))