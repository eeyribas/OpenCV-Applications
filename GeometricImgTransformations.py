import cv2
import matplotlib.pyplot as plt
import numpy as np

def ShowWithMatplotlib(img, title):
    imgRGB=img[:, :, ::-1]
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()

image=cv2.imread('images/lenna-image.png')
ShowWithMatplotlib(image, 'Original image')
dstImage1=cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

height, width=image.shape[:2]
dstImage2=cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_LINEAR)
ShowWithMatplotlib(dstImage1, 'Resized image')
ShowWithMatplotlib(dstImage2, 'Resized image 2')

M=np.float32([[1, 0, 200], [0, 1, 30]])
dstImage=cv2.warpAffine(image, M, (width, height))
ShowWithMatplotlib(dstImage, 'Translated image (positive values)')

M=np.float32([[1, 0, -200], [0, 1, -30]])
dstImage=cv2.warpAffine(image, M, (width, height))
ShowWithMatplotlib(dstImage, 'Translated image (negative values)')

M=cv2.getRotationMatrix2D((width/2.0, height/2.0), 180, 1)
dstImage=cv2.warpAffine(image, M, (width, height))
cv2.circle(dstImage, (round(width/2.0), round(height/2.0)), 5, (255, 0, 0), -1)
ShowWithMatplotlib(dstImage, 'Image rotated 180 degrees')

M=cv2.getRotationMatrix2D((width/1.5, height/1.5), 30, 1)
dstImage=cv2.warpAffine(image, M, (width, height))
cv2.circle(dstImage, (round(width/1.5), round(height/1.5)), 5, (255, 0, 0), -1)
ShowWithMatplotlib(dstImage, 'Image rotated 30 degrees')

imagePoints=image.copy()
cv2.circle(imagePoints, (135, 45), 5, (255, 0, 255), -1)
cv2.circle(imagePoints, (385, 45), 5, (255, 0, 255), -1)
cv2.circle(imagePoints, (135, 230), 5, (255, 0, 255), -1)
ShowWithMatplotlib(imagePoints, 'before affine transformation')

pts1=np.float32([[135, 45], [385, 45], [135, 230]])
pts2=np.float32([[135, 45], [385, 45], [150, 230]])
M=cv2.getAffineTransform(pts1, pts2)
dstImage=cv2.warpAffine(imagePoints, M, (width, height))
ShowWithMatplotlib(dstImage, 'Affine transformation')

imagePoints=image.copy()
cv2.circle(imagePoints, (450, 65), 5, (255, 0, 255), -1)
cv2.circle(imagePoints, (517, 65), 5, (255, 0, 255), -1)
cv2.circle(imagePoints, (431, 164), 5, (255, 0, 255), -1)
cv2.circle(imagePoints, (552, 164), 5, (255, 0, 255), -1)
ShowWithMatplotlib(imagePoints, 'before perspective transformation')

pts1=np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts2=np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M=cv2.getPerspectiveTransform(pts1, pts2)
dstImage = cv2.warpPerspective(image, M, (300, 300))
ShowWithMatplotlib(dstImage, 'perspective transformation')

imagePoints=image.copy()
cv2.circle(imagePoints, (230, 80), 5, (0, 0, 255), -1)
cv2.circle(imagePoints, (330, 80), 5, (0, 0, 255), -1)
cv2.circle(imagePoints, (230, 200), 5, (0, 0, 255), -1)
cv2.circle(imagePoints, (330, 200), 5, (0, 0, 255), -1)
cv2.line(imagePoints, (230, 80), (330, 80), (0, 0, 255))
cv2.line(imagePoints, (230, 200), (330, 200), (0, 0, 255))
cv2.line(imagePoints, (230, 80), (230, 200), (0, 0, 255))
cv2.line(imagePoints, (330, 200), (330, 80), (0, 0, 255))
ShowWithMatplotlib(imagePoints, 'Before cropping')

dstImage=image[80:200, 230:330]
ShowWithMatplotlib(dstImage, 'Cropping image')