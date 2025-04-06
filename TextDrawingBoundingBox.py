import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowWithMatplotlib(img, title):
    imgRGB=img[:, :, ::-1]
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()

colors={'blue':(255, 0, 0), 'green':(0, 255, 0), 'red':(0, 0, 255), 'yellow':(0, 255, 255),
        'magenta':(255, 0, 255), 'cyan':(255, 255, 0), 'white':(255, 255, 255), 'black':(0, 0, 0),
        'gray':(125, 125, 125), 'rand':np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray':(50, 50, 50), 'light_gray':(220, 220, 220)}

image=np.zeros((400, 1200, 3), dtype="uint8")
image[:]=colors['light_gray']
font=cv2.FONT_HERSHEY_SIMPLEX
fontScale=2.5
thickness=5
text='abcdefghijklmnopqrstuvwxyz'
circleRadius=10

ret, baseline=cv2.getTextSize(text, font, fontScale, thickness)
textWidth, textHeight=ret
textX=int(round((image.shape[1]-textWidth)/2))
textY=int(round((image.shape[0]+textHeight)/2))

cv2.circle(image, (textX, textY), circleRadius, colors['green'], -1)
cv2.rectangle(image, (textX, textY+baseline), (textX+textWidth-thickness, textY-textHeight), colors['blue'], thickness)
cv2.circle(image, (textX, textY+baseline), circleRadius, colors['red'], -1)
cv2.circle(image, (textX+textWidth-thickness, textY-textHeight), circleRadius, colors['cyan'], -1)
cv2.line(image, (textX, textY+int(round(thickness/2))), (textX+textWidth-thickness, textY+int(round(thickness/2))),
         colors['yellow'], thickness)
cv2.putText(image, text, (textX, textY), font, fontScale, colors['magenta'], thickness)
ShowWithMatplotlib(image, 'cv2.getTextSize() + cv2.putText()')