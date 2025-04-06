import numpy as np
import cv2

x=np.uint8([250])
y=np.uint8([50])
result1=cv2.add(x, y)
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, result1))
result2=x+y
print("x:'{}' + y:'{}' = '{}'".format(x, y, result2))