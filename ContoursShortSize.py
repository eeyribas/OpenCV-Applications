import cv2
from matplotlib import pyplot as plt

def GetPositionToDraw(text, point, fontFace, fontScale, thickness):
    textSize=cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    textX=point[0]-textSize[0]/2
    textY=point[1]+textSize[1]/2
    return round(textX), round(textY)

def SortContoursSize(cnts):
    cntsSizes=[cv2.contourArea(contour) for contour in cnts]
    (cntsSizes, cnts)=zip(*sorted(zip(cntsSizes, cnts)))
    return cntsSizes, cnts

def ShowImgWithMatplotlib(colorImg, title, pos):
    imgRGB=colorImg[:, :, ::-1]
    ax=plt.subplot(2, 1, pos)
    plt.imshow(imgRGB)
    plt.title(title)
    plt.axis('off')

fig=plt.figure(figsize=(9, 9))
plt.suptitle("Sort contours by size", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image=cv2.imread("images/shapes-sizes.png")
grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ShowImgWithMatplotlib(image, "image", 1)

ret, thresh=cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)
contours, hierarchy=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("detected contours: '{}' ".format(len(contours)))

(contourSizes, contours)=SortContoursSize(contours)
for i, (size, contour) in enumerate(zip(contourSizes, contours)):
    M=cv2.moments(contour)
    cX=int(M['m10']/M['m00'])
    cY=int(M['m01']/M['m00'])
    (x, y)=GetPositionToDraw(str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
    cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

ShowImgWithMatplotlib(image, "result", 2)
plt.show()